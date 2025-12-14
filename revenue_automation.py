#!/usr/bin/env python3
"""
Secure Production Revenue Automation for MyRoofGenius
All credentials loaded from environment variables
"""

import os
import time
import json
import logging
import schedule
import threading
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
import stripe
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/secure_automation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database configuration from environment
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab"),
    "password": os.getenv("DB_PASSWORD"),
    "port": os.getenv("DB_PORT", "5432")
}

# API Keys from environment
stripe.api_key = os.getenv("STRIPE_API_KEY")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
# Payment links (already created in Stripe)
PAYMENT_LINKS = {
    "basic": "https://buy.stripe.com/7sY4gz3EW5ZnfQPc6S2go04",
    "standard": "https://buy.stripe.com/5kA6pHaSa93r7ko14RX6B5",
    "premium": "https://buy.stripe.com/9AQ3dvd9qcfDbACcr3n0FH",
    "roof_inspection": "https://buy.stripe.com/eVa8xPfhyadH9ss5kLv8La",
    "maintenance": "https://buy.stripe.com/bIY9BTb1i93rcEDalBvf8S"
}

class SecureEmailAutomation:
    """Handles all email sending through SendGrid"""

    def __init__(self):
        self.sg = SendGridAPIClient(SENDGRID_API_KEY) if SENDGRID_API_KEY else None
        self.from_email = "sales@myroofgenius.com"

    def send_email(self, to_email, subject, content, html_content=None):
        """Send email via SendGrid"""
        if not self.sg:
            logger.warning("SendGrid not configured - skipping email")
            return False

        try:
            message = Mail(
                from_email=self.from_email,
                to_emails=to_email,
                subject=subject,
                plain_text_content=content,
                html_content=html_content or f"<p>{content}</p>"
            )
            response = self.sg.send(message)
            logger.info(f"Email sent to {to_email}: {response.status_code}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False

class SecureRevenueSystem:
    """Main revenue automation system"""

    def __init__(self):
        self.email = SecureEmailAutomation()
        self.ensure_tables()

    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**DB_CONFIG)

    def ensure_tables(self):
        """Create tables if they don't exist"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Revenue tracking table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS revenue_automation (
                        id SERIAL PRIMARY KEY,
                        customer_email VARCHAR(255),
                        customer_name VARCHAR(255),
                        stage VARCHAR(50),
                        payment_link VARCHAR(500),
                        amount DECIMAL(10,2),
                        status VARCHAR(50),
                        last_contact TIMESTAMP,
                        next_followup TIMESTAMP,
                        converted BOOLEAN DEFAULT FALSE,
                        stripe_customer_id VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Email log table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS email_automation_log (
                        id SERIAL PRIMARY KEY,
                        customer_email VARCHAR(255),
                        email_type VARCHAR(100),
                        subject VARCHAR(500),
                        sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        success BOOLEAN
                    )
                """)

                conn.commit()
                logger.info("Database tables ready")

    def add_lead(self, email, name, source="website"):
        """Add new lead to automation"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO revenue_automation
                    (customer_email, customer_name, stage, payment_link, amount, status, next_followup)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    RETURNING id
                """, (
                    email, name, "new_lead", PAYMENT_LINKS["basic"], 99.00,
                    "pending", datetime.now() + timedelta(hours=1)
                ))
                result = cur.fetchone()
                conn.commit()

                if result:
                    self.send_welcome_email(email, name)
                    logger.info(f"Added new lead: {email}")
                    return result[0]
                return None

    def send_welcome_email(self, email, name):
        """Send welcome email with payment link"""
        subject = "Welcome to MyRoofGenius - Start Your Roofing Business Growth"
        content = f"""
        Hi {name},

        Thank you for your interest in MyRoofGenius!

        We help roofing contractors grow their business through:
        ✅ Automated lead generation
        ✅ Smart job scheduling
        ✅ Professional estimates & invoicing
        ✅ Customer relationship management

        Get started today for just $99/month:
        {PAYMENT_LINKS["basic"]}

        Questions? Reply to this email or call us at 1-888-ROOF-AI.

        Best regards,
        The MyRoofGenius Team
        """

        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2>Welcome to MyRoofGenius, {name}!</h2>
            <p>We're excited to help you grow your roofing business.</p>
            <h3>What You Get:</h3>
            <ul>
                <li>Automated lead generation & qualification</li>
                <li>Smart scheduling & route optimization</li>
                <li>Professional estimates & invoicing</li>
                <li>Customer management & follow-ups</li>
                <li>Real-time business analytics</li>
            </ul>
            <div style="margin: 30px 0; text-align: center;">
                <a href="{PAYMENT_LINKS["basic"]}"
                   style="background: #007bff; color: white; padding: 15px 30px;
                          text-decoration: none; border-radius: 5px; display: inline-block;">
                    Start Your Free Trial - Only $99/month
                </a>
            </div>
            <p>Questions? Reply to this email or call 1-888-ROOF-AI</p>
        </body>
        </html>
        """

        self.email.send_email(email, subject, content, html)
        self.log_email(email, "welcome", subject)

    def process_followups(self):
        """Process all pending follow-ups"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM revenue_automation
                    WHERE next_followup <= NOW()
                    AND converted = FALSE
                    AND status != 'unsubscribed'
                    LIMIT 50
                """)
                leads = cur.fetchall()

                for lead in leads:
                    self.send_followup(lead)
                    self.update_next_followup(lead['id'])

        logger.info(f"Processed {len(leads)} follow-ups")

    def send_followup(self, lead):
        """Send appropriate follow-up based on stage"""
        followup_templates = {
            "new_lead": {
                "subject": "Don't Miss Out - Special Roofing Software Offer",
                "content": f"""
                Hi {lead['customer_name']},

                Just following up on MyRoofGenius. We're offering a special deal:

                🎯 50% off your first 3 months
                📈 Free data migration from your current system
                🤝 Dedicated onboarding specialist

                Claim your offer: {lead['payment_link']}

                This offer expires in 48 hours.

                Best,
                MyRoofGenius Team
                """
            },
            "trial": {
                "subject": "Your MyRoofGenius Trial is Ending Soon",
                "content": f"""
                Hi {lead['customer_name']},

                Your trial period is ending in 3 days. Don't lose access to:

                • Your saved estimates and invoices
                • Customer database
                • Scheduled jobs

                Continue with full access: {lead['payment_link']}

                Need help? Schedule a call: https://calendly.com/myroofgenius

                Best,
                MyRoofGenius Team
                """
            }
        }

        template = followup_templates.get(lead['stage'], followup_templates['new_lead'])
        self.email.send_email(
            lead['customer_email'],
            template['subject'],
            template['content']
        )
        self.log_email(lead['customer_email'], f"followup_{lead['stage']}", template['subject'])

    def update_next_followup(self, lead_id):
        """Update next follow-up time"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE revenue_automation
                    SET next_followup = NOW() + INTERVAL '2 days',
                        last_contact = NOW(),
                        updated_at = NOW()
                    WHERE id = %s
                """, (lead_id,))
                conn.commit()

    def log_email(self, email, email_type, subject):
        """Log email send"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO email_automation_log
                    (customer_email, email_type, subject, success)
                    VALUES (%s, %s, %s, %s)
                """, (email, email_type, subject, True))
                conn.commit()

    def check_stripe_payments(self):
        """Check for new payments in Stripe"""
        if not stripe.api_key:
            logger.warning("Stripe not configured")
            return

        try:
            # Get recent successful payments
            payments = stripe.PaymentIntent.list(limit=10)

            for payment in payments.data:
                if payment.status == 'succeeded':
                    # Mark as converted
                    customer_email = payment.metadata.get('email')
                    if customer_email:
                        with self.get_connection() as conn:
                            with conn.cursor() as cur:
                                cur.execute("""
                                    UPDATE revenue_automation
                                    SET converted = TRUE,
                                        status = 'customer',
                                        stripe_customer_id = %s,
                                        updated_at = NOW()
                                    WHERE customer_email = %s
                                """, (payment.customer, customer_email))
                                conn.commit()
                                logger.info(f"Customer converted: {customer_email}")
        except Exception as e:
            logger.error(f"Error checking Stripe payments: {e}")

    def get_stats(self):
        """Get current automation stats"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        COUNT(*) as total_leads,
                        SUM(CASE WHEN converted THEN 1 ELSE 0 END) as conversions,
                        SUM(CASE WHEN converted THEN amount ELSE 0 END) as revenue
                    FROM revenue_automation
                """)
                stats = cur.fetchone()

                cur.execute("""
                    SELECT COUNT(*) as emails_sent
                    FROM email_automation_log
                    WHERE sent_at > NOW() - INTERVAL '24 hours'
                """)
                email_stats = cur.fetchone()

                return {
                    "total_leads": stats['total_leads'],
                    "conversions": stats['conversions'],
                    "revenue": float(stats['revenue'] or 0),
                    "emails_24h": email_stats['emails_sent']
                }

def run_automation():
    """Main automation loop"""
    system = SecureRevenueSystem()

    # Schedule tasks
    schedule.every(30).minutes.do(system.process_followups)
    schedule.every(1).hours.do(system.check_stripe_payments)

    # Add some test leads if none exist
    stats = system.get_stats()
    if stats['total_leads'] == 0:
        logger.info("Adding test leads...")
        system.add_lead("john@roofingpro.com", "John Smith", "website")
        system.add_lead("mary@qualityroofs.com", "Mary Johnson", "google")
        system.add_lead("bob@topnotchroofing.com", "Bob Williams", "referral")

    logger.info(f"Starting automation - Current stats: {stats}")

    # Run scheduled tasks
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    # Verify environment variables
    if not os.getenv("DB_PASSWORD"):
        logger.error("DB_PASSWORD not set in environment")
    if not os.getenv("STRIPE_API_KEY"):
        logger.warning("STRIPE_API_KEY not set - payment tracking disabled")
    if not os.getenv("SENDGRID_API_KEY"):
        logger.warning("SENDGRID_API_KEY not set - emails disabled")

    try:
        run_automation()
    except KeyboardInterrupt:
        logger.info("Automation stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)