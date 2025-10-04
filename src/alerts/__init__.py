"""
Alert and notification system for portfolio monitoring.

This package provides email and SMS alerts for critical portfolio events,
risk threshold breaches, and system errors.

Supported Channels:
- Email (SMTP/Gmail/SendGrid)
- SMS (Twilio)
- Slack (via webhooks)
"""

from .alert_manager import AlertManager, AlertRule, AlertLevel
from .email_sender import EmailSender
from .sms_sender import SMSSender

__all__ = [
    'AlertManager',
    'AlertRule',
    'AlertLevel',
    'EmailSender',
    'SMSSender'
]
