#!/usr/bin/env python3
"""
Lead Gen Segmentation Sequences
=============================================
Automated follow-up sequences for 'Reroof Genius' and 'Service Genius' segments.

Segments:
1. Reroof Genius: Residential replacement focus (Storm, Age, Insurance)
2. Service Genius: Repair & Maintenance focus (Leaks, Tune-ups)

These sequences are triggered based on the lead source/packet type.
"""

import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import psycopg2
from psycopg2.extras import Json

logger = logging.getLogger(__name__)


def _is_test_email(email: str | None) -> bool:
    if not email:
        return True
    lowered = email.strip().lower()
    return any(
        token in lowered
        for token in ("test", "example", "demo", "sample", "fake", "placeholder", "localhost")
    )

# Reroof Genius Sequence
REROOF_GENIUS_SEQUENCE = {
    "name": "Reroof Genius - Residential Replacement",
    "sequence_type": "sales",
    "target_segment": "reroof",
    "touchpoints": [
        {
            "touchpoint_number": 1,
            "days_after_trigger": 0,
            "time_of_day": "09:00:00",
            "touchpoint_type": "email",
            "subject_line": "Your Roof Replacement Estimate Request - Reroof Genius",
            "content_template": """Hi {first_name},

Thanks for choosing Reroof Genius. We received your request for a roof replacement estimate for your home in {city}.

Based on your home's profile ({roof_age_years} years old), it sounds like a full replacement might be the right move.

What happens next:
1. We'll do a remote satellite analysis of your roof first.
2. We'll contact you within 24 hours to schedule a quick 15-minute on-site verification.
3. You'll get a guaranteed price - not just an estimate.

If this is related to an insurance claim, please let us know when we speak.

Talk soon,
The Reroof Genius Team
""",
            "call_to_action": "Reply if urgent",
        },
        {
            "touchpoint_number": 2,
            "days_after_trigger": 2,
            "time_of_day": "10:00:00",
            "touchpoint_type": "email",
            "subject_line": "3 Signs Your Roof Needs Replacement (Not Repair)",
            "content_template": """Hi {first_name},

Since you're considering a new roof, here are the 3 signs we look for that indicate replacement is better than repair:

1. Granule Loss: If your shingles look like they are balding, the UV protection is gone.
2. Brittle Shingles: If they crack when touched, they can't be repaired.
3. Age: Once asphalt shingles pass 15-20 years in our climate, they lose their waterproof seal.

Our Reroof Genius assessment checks exactly for these. We don't want to sell you a roof you don't need, but we also don't want you wasting money repairing a dying roof.

Have you scheduled your assessment yet?

Best,
The Reroof Genius Team
""",
            "call_to_action": "Schedule Assessment",
        },
        {
            "touchpoint_number": 3,
            "days_after_trigger": 5,
            "time_of_day": "15:00:00",
            "touchpoint_type": "email",
            "subject_line": "Did you know about our 'No-Mess' Guarantee?",
            "content_template": """Hi {first_name},

One big worry homeowners have about reroofing is the mess. Nails in the driveway, debris in the bushes.

At Reroof Genius, we have a "No-Mess" Guarantee. We use the Catch-All landscape protection system and run magnetic sweeps 3 times during the job.

If you find a nail after we leave, we pay you $1 per nail (up to $100).

Ready to get that new roof with zero headaches?

Reply 'READY' to book your slot.

Best,
The Reroof Genius Team
""",
            "call_to_action": "Reply READY",
        }
    ]
}

# Service Genius Sequence
SERVICE_GENIUS_SEQUENCE = {
    "name": "Service Genius - Repair & Maintenance",
    "sequence_type": "service",
    "target_segment": "repair",
    "touchpoints": [
        {
            "touchpoint_number": 1,
            "days_after_trigger": 0,
            "time_of_day": "08:30:00",
            "touchpoint_type": "email",
            "subject_line": "We received your repair request - Service Genius",
            "content_template": """Hi {first_name},

We got your request about a roof issue at your property in {city}.

Our Service Genius technicians are experts at finding and fixing leaks, not just selling new roofs.

We'll be reaching out shortly to get a few more details and get a truck headed your way.

If this is an active leak causing interior damage right now, please call our emergency line at (555) 123-4567 immediately.

Stay dry,
The Service Genius Team
""",
            "call_to_action": "Call if emergency",
        },
        {
            "touchpoint_number": 2,
            "days_after_trigger": 3,
            "time_of_day": "09:00:00",
            "touchpoint_type": "email",
            "subject_line": "Why we recommend a 'Roof Tune-Up'",
            "content_template": """Hi {first_name},

Even if you just have one leak, we often recommend our Service Genius 'Roof Tune-Up'.

For a flat fee, we:
1. Seal all pipe jacks (the #1 cause of leaks)
2. Reseal flashings and vents
3. Nail down pop-ups
4. Clear minor debris from valleys

It's like an oil change for your roof. It prevents the *next* leak.

Ask your technician about the Tune-Up special when they arrive.

Best,
The Service Genius Team
""",
            "call_to_action": "Ask about Tune-Up",
        },
        {
            "touchpoint_number": 3,
            "days_after_trigger": 14,
            "time_of_day": "11:00:00",
            "touchpoint_type": "email",
            "subject_line": "Checking in on your repair",
            "content_template": """Hi {first_name},

Just wanted to check in - has your roof issue been resolved to your satisfaction?

If we've completed the work, we hope everything is dry and secure.

If you haven't scheduled yet, please let us know. Small leaks turn into big rot if left too long!

Reply 'STILL LEAKING' if you need us back out.

Best,
The Service Genius Team
""",
            "call_to_action": "Reply with status",
        }
    ]
}

def _get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=int(os.getenv("DB_PORT", "5432")),
    )

def enroll_lead_in_segment_sequence(
    lead_id: str,
    email: str,
    first_name: str,
    segment: str,  # 'reroof' or 'service'
    lead_data: Dict[str, Any]
) -> bool:
    """
    Enroll a lead in the appropriate segment sequence.
    """
    try:
        if segment == 'reroof':
            sequence_def = REROOF_GENIUS_SEQUENCE
        elif segment == 'service':
            sequence_def = SERVICE_GENIUS_SEQUENCE
        else:
            logger.warning(f"Unknown segment: {segment}")
            return False

        conn = _get_db_connection()
        cursor = conn.cursor()

        # 1. Ensure Sequence Exists
        cursor.execute("SELECT id FROM ai_nurture_sequences WHERE name = %s", (sequence_def['name'],))
        row = cursor.fetchone()
        
        now = datetime.now(timezone.utc)
        
        if row:
            sequence_id = row[0]
        else:
            sequence_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO ai_nurture_sequences 
                (id, name, sequence_type, target_segment, is_active, configuration)
                VALUES (%s, %s, %s, %s, true, %s)
            """, (
                sequence_id, 
                sequence_def['name'], 
                sequence_def['sequence_type'], 
                sequence_def['target_segment'],
                Json({"created_at": now.isoformat()})
            ))
            
            for tp in sequence_def['touchpoints']:
                cursor.execute("""
                    INSERT INTO ai_sequence_touchpoints
                    (id, sequence_id, touchpoint_number, touchpoint_type, days_after_trigger, time_of_day, subject_line, content_template, call_to_action)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    str(uuid.uuid4()), sequence_id, tp['touchpoint_number'], tp['touchpoint_type'],
                    tp['days_after_trigger'], tp['time_of_day'], tp['subject_line'], tp['content_template'], tp['call_to_action']
                ))

        # 2. Check Enrollment
        cursor.execute("SELECT id FROM ai_lead_enrollments WHERE lead_id = %s AND sequence_id = %s", (lead_id, sequence_id))
        if cursor.fetchone():
            logger.info(f"Lead {lead_id} already enrolled in {segment} sequence")
            conn.close()
            return True

        # 3. Enroll
        enrollment_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO ai_lead_enrollments (id, lead_id, sequence_id, enrollment_date, status, metadata)
            VALUES (%s, %s, %s, %s, 'active', %s)
        """, (enrollment_id, lead_id, sequence_id, now, Json(lead_data)))

        # 4. Schedule Emails
        cursor.execute("SELECT id, touchpoint_number, days_after_trigger, time_of_day, subject_line, content_template FROM ai_sequence_touchpoints WHERE sequence_id = %s", (sequence_id,))
        touchpoints = cursor.fetchall()
        
        for tp in touchpoints:
            tp_id, tp_num, days, time_of_day, subject, body = tp
            
            # Simple Personalization
            personalized_subject = subject.replace("{first_name}", first_name).replace("{city}", lead_data.get('city', 'your area')).replace("{roof_age_years}", str(lead_data.get('roof_age_years', 'unknown')))
            personalized_body = body.replace("{first_name}", first_name).replace("{city}", lead_data.get('city', 'your area')).replace("{roof_age_years}", str(lead_data.get('roof_age_years', 'unknown')))
            
            scheduled_date = now + timedelta(days=days)
            # (Time logic simplified for brevity)

            cursor.execute("""
                INSERT INTO ai_email_queue (id, recipient, subject, body, scheduled_for, status, metadata)
                VALUES (%s, %s, %s, %s, %s, 'queued', %s)
            """, (
                str(uuid.uuid4()), email, personalized_subject, personalized_body, scheduled_date,
                Json(
                    {
                        "enrollment_id": enrollment_id,
                        "sequence": segment,
                        "segment": segment,
                        "lead_id": lead_id,
                        "source": "lead_segmentation_sequence",
                        "touchpoint_number": tp_num,
                        "is_test": _is_test_email(email),
                    }
                ),
            ))

        conn.commit()
        conn.close()
        logger.info(f"Enrolled {lead_id} in {segment} sequence")
        return True

    except Exception as e:
        logger.error(f"Failed to enroll in segment sequence: {e}")
        return False
