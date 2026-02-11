"""
Gumroad Post-Purchase Email Sequences
=====================================
Pre-built nurture sequences for Gumroad product buyers.
Triggered automatically when a purchase is made via webhook.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Email sequence definitions for each product type
PRODUCT_SEQUENCES = {
    "code_kit": {
        "name": "Code Kit Onboarding",
        "emails": [
            {
                "delay_minutes": 0,
                "subject": "Your {product_name} is ready - Let's get started!",
                "body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2>Hey {first_name}!</h2>

    <p>Thanks for purchasing <strong>{product_name}</strong>! You've made a great decision.</p>

    <p><a href="{download_url}" style="background: #4CAF50; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block;">Download Your Files</a></p>

    <h3>Quick Start Guide:</h3>
    <ol>
        <li>Download and unzip the package</li>
        <li>Read the README.md for setup instructions</li>
        <li>Run the example code to see it in action</li>
    </ol>

    <p>If you have any questions, just reply to this email!</p>

    <p>Best,<br>Matt @ BrainStack</p>
</div>
"""
            },
            {
                "delay_minutes": 1440,  # Day 1
                "subject": "Quick tip for {product_name}",
                "body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2>Hey {first_name}!</h2>

    <p>Just checking in - did you get a chance to set up <strong>{product_name}</strong>?</p>

    <p><strong>Pro tip:</strong> Start with the example files first. They show you the patterns without having to figure everything out from scratch.</p>

    <p>Most customers who get stuck do so because they try to customize before understanding the base implementation. The examples are your friend!</p>

    <p>Need help? Just reply to this email - I read every one.</p>

    <p>Best,<br>Matt</p>
</div>
"""
            },
            {
                "delay_minutes": 4320,  # Day 3
                "subject": "How's it going with {product_name}?",
                "body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2>Hey {first_name}!</h2>

    <p>You've had <strong>{product_name}</strong> for a few days now. How's it going?</p>

    <p>I'd love to hear:</p>
    <ul>
        <li>What you're building with it</li>
        <li>Any questions you've run into</li>
        <li>Features you wish it had</li>
    </ul>

    <p>Your feedback helps me make better products. Just hit reply!</p>

    <p>Best,<br>Matt</p>
</div>
"""
            },
            {
                "delay_minutes": 10080,  # Day 7
                "subject": "One week with {product_name} - any wins to share?",
                "body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2>Hey {first_name}!</h2>

    <p>It's been a week since you got <strong>{product_name}</strong>.</p>

    <p>If you've shipped something using it, I'd love to hear about it! Customer success stories help other developers know what's possible.</p>

    <p>And if you're happy with your purchase, a quick review on Gumroad would mean a lot. It helps other developers find these tools.</p>

    <p>Thanks for being a customer!</p>

    <p>Best,<br>Matt</p>
</div>
"""
            }
        ]
    },
    "prompt_pack": {
        "name": "Prompt Pack Onboarding",
        "emails": [
            {
                "delay_minutes": 0,
                "subject": "Your {product_name} is ready!",
                "body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2>Hey {first_name}!</h2>

    <p>Thanks for purchasing <strong>{product_name}</strong>!</p>

    <p><a href="{download_url}" style="background: #4CAF50; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block;">Download Your Prompts</a></p>

    <h3>How to get the most value:</h3>
    <ol>
        <li>Browse the categories to find prompts for your use case</li>
        <li>Copy/paste into ChatGPT, Claude, or your favorite AI</li>
        <li>Customize the variables in [brackets] for your context</li>
    </ol>

    <p>These prompts are tested and refined - they'll save you hours of trial and error!</p>

    <p>Best,<br>Matt @ BrainStack</p>
</div>
"""
            },
            {
                "delay_minutes": 1440,  # Day 1
                "subject": "My favorite prompt from {product_name}",
                "body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2>Hey {first_name}!</h2>

    <p>Quick tip: The most underrated prompts in <strong>{product_name}</strong> are usually in the "edge cases" or "advanced" sections.</p>

    <p>Most people only use the obvious ones. But the real magic happens when you chain prompts together or use them for tasks you didn't expect.</p>

    <p>Try this: Pick a prompt from a category you wouldn't normally use. You might be surprised what it can do for your workflow.</p>

    <p>Best,<br>Matt</p>
</div>
"""
            },
            {
                "delay_minutes": 4320,  # Day 3
                "subject": "Quick question about {product_name}",
                "body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2>Hey {first_name}!</h2>

    <p>Have you found your go-to prompts yet from <strong>{product_name}</strong>?</p>

    <p>I'm always looking to improve these collections. If there's a use case you wish was covered, let me know and I might add it to the next update!</p>

    <p>Just hit reply - I read every email.</p>

    <p>Best,<br>Matt</p>
</div>
"""
            },
            {
                "delay_minutes": 10080,  # Day 7
                "subject": "Happy with {product_name}?",
                "body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2>Hey {first_name}!</h2>

    <p>It's been a week with <strong>{product_name}</strong>.</p>

    <p>If you're getting value from it, I'd really appreciate a quick review on Gumroad. It helps other people find these resources!</p>

    <p>And if you have any prompt requests for future updates, I'm all ears.</p>

    <p>Thanks for your support!</p>

    <p>Best,<br>Matt</p>
</div>
"""
            }
        ]
    },
    "bundle": {
        "name": "Bundle VIP Onboarding",
        "emails": [
            {
                "delay_minutes": 0,
                "subject": "Welcome to the {product_name} - You're a VIP!",
                "body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2>Hey {first_name}!</h2>

    <p>WOW - thank you for getting the <strong>{product_name}</strong>!</p>

    <p><a href="{download_url}" style="background: #4CAF50; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block;">Access All Your Products</a></p>

    <p>As a bundle customer, you're a VIP. That means:</p>
    <ul>
        <li>Priority email support (I respond to bundle customers first)</li>
        <li>Access to all future updates</li>
        <li>First access to new products</li>
    </ul>

    <p>If you ever need anything, just reply to this email. You're at the front of the line.</p>

    <p>Best,<br>Matt @ BrainStack</p>
</div>
"""
            },
            {
                "delay_minutes": 1440,  # Day 1
                "subject": "Where to start with your bundle",
                "body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2>Hey {first_name}!</h2>

    <p>With so much in the bundle, you might be wondering where to start.</p>

    <p><strong>My recommendation:</strong> Pick the ONE tool that solves your most pressing problem right now. Master that first, then expand.</p>

    <p>Don't try to learn everything at once - that's a recipe for overwhelm. The bundle is a toolkit, not a curriculum.</p>

    <p>What's your #1 priority right now? I can point you to the right place to start.</p>

    <p>Best,<br>Matt</p>
</div>
"""
            },
            {
                "delay_minutes": 4320,  # Day 3
                "subject": "VIP check-in",
                "body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2>Hey {first_name}!</h2>

    <p>Quick VIP check-in - how's everything going with your bundle?</p>

    <p>As a bundle customer, I want to make sure you're getting maximum value. If there's anything confusing or you're stuck anywhere, let me know.</p>

    <p>Your success is my success!</p>

    <p>Best,<br>Matt</p>
</div>
"""
            },
            {
                "delay_minutes": 10080,  # Day 7
                "subject": "One week as a VIP - how can I help?",
                "body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2>Hey {first_name}!</h2>

    <p>It's been a week since you joined as a VIP bundle customer.</p>

    <p>I'd love to hear what you've built or accomplished so far. And if there's anything I can do to help you succeed, I'm here.</p>

    <p>Also, if you're happy with your purchase, a Gumroad review would mean a lot. Bundle customers are my best advocates!</p>

    <p>Best,<br>Matt</p>
</div>
"""
            }
        ]
    }
}


async def enroll_buyer_in_sequence(
    email: str,
    first_name: str,
    product_name: str,
    product_type: str,
    product_code: str,
    download_url: str,
    sale_id: Optional[str] = None,
) -> list[str]:
    """
    Enroll a Gumroad buyer in the appropriate post-purchase email sequence.
    Returns list of scheduled email IDs.
    """
    try:
        from email_scheduler_daemon import schedule_nurture_email

        # Get sequence template for product type
        sequence_config = PRODUCT_SEQUENCES.get(product_type, PRODUCT_SEQUENCES["code_kit"])

        email_ids = []

        # Idempotency: Gumroad can deliver duplicate webhook events (retries, multiple subscriptions).
        # Deduplicate scheduled emails per sale/product/delay so we don't spam buyers.
        from database.async_connection import get_pool
        pool = get_pool()
        sale_id_value = (sale_id or "").strip()
        async with pool.acquire() as conn:
            for email_template in sequence_config["emails"]:
                delay_minutes = int(email_template.get("delay_minutes", 0) or 0)

                # Personalize email content
                subject = email_template["subject"].format(
                    first_name=first_name or "there",
                    product_name=product_name
                )

                body = email_template["body"].format(
                    first_name=first_name or "there",
                    product_name=product_name,
                    download_url=download_url or "https://gumroad.com/library"
                )

                exists = await conn.fetchval(
                    """
                    SELECT 1
                    FROM ai_email_queue
                    WHERE recipient = $1
                      AND subject = $2
                      AND COALESCE(metadata->>'source', '') = 'gumroad_sequence'
                      AND COALESCE(metadata->>'product_code', '') = $3
                      AND COALESCE((metadata->>'delay_minutes')::int, -1) = $4
                      AND ($5 = '' OR COALESCE(metadata->>'sale_id', '') = $5)
                    LIMIT 1
                    """,
                    email,
                    subject,
                    product_code,
                    delay_minutes,
                    sale_id_value,
                )

                if exists:
                    logger.info(
                        "Skipping duplicate nurture email (sale_id=%s product_code=%s delay=%s recipient=%s)",
                        sale_id_value or "<missing>",
                        product_code,
                        delay_minutes,
                        email,
                    )
                    continue

                # Schedule the email
                email_id = await schedule_nurture_email(
                    recipient=email,
                    subject=subject,
                    body=body,
                    delay_minutes=delay_minutes,
                    metadata={
                        "source": "gumroad_sequence",
                        "sale_id": sale_id_value or None,
                        "product_code": product_code,
                        "product_name": product_name,
                        "product_type": product_type,
                        "sequence_name": sequence_config["name"],
                        "delay_minutes": delay_minutes
                    }
                )

                if email_id:
                    email_ids.append(email_id)
                    logger.info(f"Scheduled sequence email for {email}: delay={delay_minutes}min")

        logger.info(f"Enrolled {email} in {sequence_config['name']} sequence: {len(email_ids)} emails scheduled")
        return email_ids

    except Exception as e:
        logger.error(f"Failed to enroll buyer in sequence: {e}")
        return []


async def get_buyer_sequence_status(email: str) -> dict:
    """Get the sequence status for a buyer"""
    try:
        from database.async_connection import get_pool
        pool = get_pool()

        rows = await pool.fetch("""
            SELECT id, subject, scheduled_for, status, metadata, sent_at
            FROM ai_email_queue
            WHERE recipient = $1
              AND metadata->>'source' = 'gumroad_sequence'
            ORDER BY scheduled_for ASC
        """, email)

        return {
            "recipient": email,
            "emails": [dict(row) for row in rows],
            "total": len(rows),
            "sent": len([r for r in rows if r["status"] == "sent"]),
            "pending": len([r for r in rows if r["status"] in ("queued", "scheduled")])
        }

    except Exception as e:
        logger.error(f"Failed to get buyer sequence status: {e}")
        return {"error": str(e)}
