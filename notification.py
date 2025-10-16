"""
Notification module for sending job status notifications via Azure Communication Service
"""
from loguru import logger
from azure.communication.email import EmailClient
from typing import Any, Dict
from config import settings


def send_email_notification(job_status: Dict[str, Any]) -> bool:
    """Send email notification via Azure Communication Service.

    Args:
        job_status: Dictionary containing job execution status and metadata
            Required keys:
                - status: Job status ('success' or 'failed')
                - job_name: Internal job identifier
                - job_display_name: Human-readable job name
                - start_time: ISO format start time
                - end_time: ISO format end time
                - duration_seconds: Duration in seconds
            Optional keys:
                - error: Error message (if failed)
                - warning: Warning message
                - results: Dict with execution results

    Returns:
        True if email sent successfully, False otherwise.
    """
    try:
        # Check if ACS is configured using the new helper method
        if not settings.is_email_configured():
            logger.warning("Azure Communication Service not configured, skipping email notification")
            return False

        # Get validated recipient emails
        to_emails = settings.get_to_emails_list()
        if not to_emails:
            logger.warning("No recipient emails configured, skipping email notification")
            return False

        # Prepare email content
        status = job_status.get('status', 'unknown')
        # Separate internal name (job_name) and display name (job_display_name)
        job_internal_name = job_status.get('job_name', 'update_daily_data')
        job_display_name = job_status.get('job_display_name') or job_status.get('job_name') or 'Daily Stock Data Update'
        start_time = job_status.get('start_time')
        end_time = job_status.get('end_time')
        duration = job_status.get('duration_seconds', 0)
        
        # Calculate duration in readable format
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        duration_str = f"{hours}h {minutes}m {seconds}s" if hours > 0 else f"{minutes}m {seconds}s"

        # Determine subject based on status
        if status == 'success':
            subject = f"✅ {job_display_name} - Success"
        elif status == 'failed':
            subject = f"❌ {job_display_name} - Failed"
        else:
            subject = f"⚠️ {job_display_name} - {status.capitalize()}"

        # Build email body
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <h2 style="color: {'#28a745' if status == 'success' else '#dc3545'};">{subject}</h2>
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0;">
                <h3 style="margin-top: 0;">Job Details</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 8px 0;"><strong>Job:</strong></td>
                        <td style="padding: 8px 0;">{job_display_name} <span style="color:#888; font-size:0.85em;">({job_internal_name})</span></td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0;"><strong>Status:</strong></td>
                        <td style="padding: 8px 0; color: {'#28a745' if status == 'success' else '#dc3545'}; font-weight: bold;">
                            {status.upper()}
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0;"><strong>Start Time:</strong></td>
                        <td style="padding: 8px 0;">{start_time}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0;"><strong>End Time:</strong></td>
                        <td style="padding: 8px 0;">{end_time}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0;"><strong>Duration:</strong></td>
                        <td style="padding: 8px 0;">{duration_str}</td>
                    </tr>
        """

        # Add error information if present
        if 'error' in job_status:
            html_body += f"""
                    <tr>
                        <td colspan="2" style="padding: 15px 0 8px 0;"><strong style="color: #dc3545;">Error:</strong></td>
                    </tr>
                    <tr>
                        <td colspan="2" style="padding: 4px 0 4px 20px; color: #dc3545;">
                            {job_status['error']}
                        </td>
                    </tr>
            """

        # Add warning information if present
        if 'warning' in job_status:
            html_body += f"""
                    <tr>
                        <td colspan="2" style="padding: 15px 0 8px 0;"><strong style="color: #ff9800;">Warning:</strong></td>
                    </tr>
                    <tr>
                        <td colspan="2" style="padding: 4px 0 4px 20px; color: #ff9800;">
                            {job_status['warning']}
                        </td>
                    </tr>
            """

        html_body += """
                </table>
            </div>
            <p style="color: #6c757d; font-size: 0.9em; margin-top: 30px;">
                This is an automated notification from QStock Collector.
            </p>
        </body>
        </html>
        """

        # Create and send email
        try:
            client = EmailClient.from_connection_string(settings.acs_connection_string)
        except Exception as e:
            logger.error(f"Failed to create email client: {e}")
            return False
        
        message = {
            "senderAddress": settings.acs_sender_email,
            "recipients": {
                "to": [{"address": email} for email in to_emails]
            },
            "content": {
                "subject": subject,
                "html": html_body
            }
        }

        try:
            poller = client.begin_send(message)
            result = poller.result()
            
            logger.info(f"Email notification sent successfully. Message ID: {result['id']}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email via ACS: {e}")
            return False

    except Exception as e:
        logger.error(f"Failed to send email notification: {e}")
        return False
