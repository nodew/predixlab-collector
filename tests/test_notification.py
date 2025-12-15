"""Unit tests for notification.py module."""
from unittest.mock import patch, MagicMock


class TestSendEmailNotification:
    """Tests for send_email_notification function."""

    def test_returns_false_when_email_not_configured(self):
        """Test that False is returned when email is not configured."""
        with patch("notification.settings") as mock_settings:
            mock_settings.is_email_configured.return_value = False

            from notification import send_email_notification

            job_status = {
                "status": "success",
                "job_name": "test_job",
                "job_display_name": "Test Job",
                "start_time": "2024-01-15T10:00:00",
                "end_time": "2024-01-15T10:30:00",
                "duration_seconds": 1800
            }

            result = send_email_notification(job_status)
            assert result is False

    def test_returns_false_when_no_recipient_emails(self):
        """Test that False is returned when no recipient emails configured."""
        with patch("notification.settings") as mock_settings:
            mock_settings.is_email_configured.return_value = True
            mock_settings.get_to_emails_list.return_value = []

            from notification import send_email_notification

            job_status = {
                "status": "success",
                "job_name": "test_job",
                "job_display_name": "Test Job",
                "start_time": "2024-01-15T10:00:00",
                "end_time": "2024-01-15T10:30:00",
                "duration_seconds": 1800
            }

            result = send_email_notification(job_status)
            assert result is False

    def test_sends_email_successfully(self):
        """Test successful email sending."""
        with patch("notification.settings") as mock_settings, \
             patch("notification.EmailClient") as mock_email_client_class:

            # Configure mock settings
            mock_settings.is_email_configured.return_value = True
            mock_settings.get_to_emails_list.return_value = ["user@example.com"]
            mock_settings.acs_connection_string = "connection-string"
            mock_settings.acs_sender_email = "sender@example.com"

            # Configure mock email client
            mock_client = MagicMock()
            mock_poller = MagicMock()
            mock_poller.result.return_value = {"id": "message-123"}
            mock_client.begin_send.return_value = mock_poller
            mock_email_client_class.from_connection_string.return_value = mock_client

            from notification import send_email_notification

            job_status = {
                "status": "success",
                "job_name": "test_job",
                "job_display_name": "Test Job",
                "start_time": "2024-01-15T10:00:00",
                "end_time": "2024-01-15T10:30:00",
                "duration_seconds": 1800
            }

            result = send_email_notification(job_status)
            assert result is True
            mock_client.begin_send.assert_called_once()

    def test_returns_false_when_email_client_creation_fails(self):
        """Test that False is returned when email client creation fails."""
        with patch("notification.settings") as mock_settings, \
             patch("notification.EmailClient") as mock_email_client_class:

            mock_settings.is_email_configured.return_value = True
            mock_settings.get_to_emails_list.return_value = ["user@example.com"]
            mock_settings.acs_connection_string = "connection-string"
            mock_email_client_class.from_connection_string.side_effect = Exception("Connection failed")

            from notification import send_email_notification

            job_status = {
                "status": "success",
                "job_name": "test_job",
                "job_display_name": "Test Job",
                "start_time": "2024-01-15T10:00:00",
                "end_time": "2024-01-15T10:30:00",
                "duration_seconds": 1800
            }

            result = send_email_notification(job_status)
            assert result is False

    def test_returns_false_when_send_fails(self):
        """Test that False is returned when email sending fails."""
        with patch("notification.settings") as mock_settings, \
             patch("notification.EmailClient") as mock_email_client_class:

            mock_settings.is_email_configured.return_value = True
            mock_settings.get_to_emails_list.return_value = ["user@example.com"]
            mock_settings.acs_connection_string = "connection-string"
            mock_settings.acs_sender_email = "sender@example.com"

            mock_client = MagicMock()
            mock_client.begin_send.side_effect = Exception("Send failed")
            mock_email_client_class.from_connection_string.return_value = mock_client

            from notification import send_email_notification

            job_status = {
                "status": "success",
                "job_name": "test_job",
                "job_display_name": "Test Job",
                "start_time": "2024-01-15T10:00:00",
                "end_time": "2024-01-15T10:30:00",
                "duration_seconds": 1800
            }

            result = send_email_notification(job_status)
            assert result is False

    def test_handles_failed_status(self):
        """Test email generation for failed job status."""
        with patch("notification.settings") as mock_settings, \
             patch("notification.EmailClient") as mock_email_client_class:

            mock_settings.is_email_configured.return_value = True
            mock_settings.get_to_emails_list.return_value = ["user@example.com"]
            mock_settings.acs_connection_string = "connection-string"
            mock_settings.acs_sender_email = "sender@example.com"

            mock_client = MagicMock()
            mock_poller = MagicMock()
            mock_poller.result.return_value = {"id": "message-123"}
            mock_client.begin_send.return_value = mock_poller
            mock_email_client_class.from_connection_string.return_value = mock_client

            from notification import send_email_notification

            job_status = {
                "status": "failed",
                "job_name": "test_job",
                "job_display_name": "Test Job",
                "start_time": "2024-01-15T10:00:00",
                "end_time": "2024-01-15T10:30:00",
                "duration_seconds": 1800,
                "error": "Something went wrong"
            }

            result = send_email_notification(job_status)
            assert result is True

            # Verify the message contains error information
            call_args = mock_client.begin_send.call_args
            message = call_args[0][0]
            assert "Failed" in message["content"]["subject"]

    def test_handles_warning_in_status(self):
        """Test email generation includes warning information."""
        with patch("notification.settings") as mock_settings, \
             patch("notification.EmailClient") as mock_email_client_class:

            mock_settings.is_email_configured.return_value = True
            mock_settings.get_to_emails_list.return_value = ["user@example.com"]
            mock_settings.acs_connection_string = "connection-string"
            mock_settings.acs_sender_email = "sender@example.com"

            mock_client = MagicMock()
            mock_poller = MagicMock()
            mock_poller.result.return_value = {"id": "message-123"}
            mock_client.begin_send.return_value = mock_poller
            mock_email_client_class.from_connection_string.return_value = mock_client

            from notification import send_email_notification

            job_status = {
                "status": "success",
                "job_name": "test_job",
                "job_display_name": "Test Job",
                "start_time": "2024-01-15T10:00:00",
                "end_time": "2024-01-15T10:30:00",
                "duration_seconds": 1800,
                "warning": "DB save failed"
            }

            result = send_email_notification(job_status)
            assert result is True

    def test_duration_formatting_hours(self):
        """Test duration formatting for hours."""
        with patch("notification.settings") as mock_settings, \
             patch("notification.EmailClient") as mock_email_client_class:

            mock_settings.is_email_configured.return_value = True
            mock_settings.get_to_emails_list.return_value = ["user@example.com"]
            mock_settings.acs_connection_string = "connection-string"
            mock_settings.acs_sender_email = "sender@example.com"

            mock_client = MagicMock()
            mock_poller = MagicMock()
            mock_poller.result.return_value = {"id": "message-123"}
            mock_client.begin_send.return_value = mock_poller
            mock_email_client_class.from_connection_string.return_value = mock_client

            from notification import send_email_notification

            # Duration of 1 hour 30 minutes 45 seconds
            job_status = {
                "status": "success",
                "job_name": "test_job",
                "job_display_name": "Test Job",
                "start_time": "2024-01-15T10:00:00",
                "end_time": "2024-01-15T11:30:45",
                "duration_seconds": 5445
            }

            result = send_email_notification(job_status)
            assert result is True

            # Verify duration is formatted with hours
            call_args = mock_client.begin_send.call_args
            message = call_args[0][0]
            assert "1h 30m 45s" in message["content"]["html"]
