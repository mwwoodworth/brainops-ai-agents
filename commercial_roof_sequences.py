"""
Commercial Roof Lead Engine Nurture Sequences
=============================================
Automated follow-up sequences for leads from MyRoofGenius Roof Health Advisory.

Sequence: commercial_roof_advisory
- Day 0: Immediate acknowledgment + next steps
- Day 1: Education about their roof type (TPO/EPDM/PVC/etc)
- Day 3: Case study for similar commercial projects
- Day 7: Gentle follow-up if not yet scheduled
- Day 14: Final touchpoint with limited-time offer

These are enrolled automatically when a lead opts in via the Sacred Handoff.
"""

import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import psycopg2
from psycopg2.extras import Json

logger = logging.getLogger(__name__)

# Test email filtering
TEST_PATTERNS = ["@example.", "@test.", "@demo.", "@invalid.", "+test"]


def _is_test_email(email: str | None) -> bool:
    if not email:
        return True
    lowered = email.lower().strip()
    return any(p in lowered for p in TEST_PATTERNS)


def _get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=int(os.getenv("DB_PORT", "5432")),
    )


# Commercial Roof Advisory Sequence Definition
COMMERCIAL_ROOF_SEQUENCE = {
    "name": "Commercial Roof Advisory - Post-Handoff",
    "sequence_type": "onboarding",
    "target_segment": "qualified",
    "touchpoints": [
        {
            "touchpoint_number": 1,
            "days_after_trigger": 0,
            "time_of_day": "09:00:00",
            "touchpoint_type": "email",
            "subject_line": "Your Free Commercial Roof Assessment is Confirmed",
            "content_template": """Hi {first_name},

Thank you for requesting a professional commercial roof assessment through MyRoofGenius.

Based on your {roof_type} system ({square_footage} sq ft), we've identified some key areas our certified inspector will evaluate:

{primary_concern_insight}

What's Next:
1. Our team will contact you within 24 hours via {preferred_contact_method}
2. We'll schedule a convenient time for your free on-site assessment
3. You'll receive a detailed report with photos and recommendations

If you have questions before then, simply reply to this email.

Best regards,
Weathercraft Roofing
Certified Commercial Roofing Specialists
""",
            "call_to_action": "Reply with any questions",
        },
        {
            "touchpoint_number": 2,
            "days_after_trigger": 1,
            "time_of_day": "10:00:00",
            "touchpoint_type": "email",
            "subject_line": "Understanding Your {roof_type} Roof System",
            "content_template": """Hi {first_name},

As we prepare for your commercial roof assessment, here's what you should know about your {roof_type} system:

{roof_type_education}

Common Issues We'll Inspect:
• Membrane condition and seam integrity
• Drainage and ponding areas
• Flashing and penetration seals
• Insulation R-value efficiency

Our inspector has completed over 500 commercial roof assessments in your area. We'll give you an honest evaluation - even if your roof doesn't need immediate work.

Talk soon,
Weathercraft Roofing
""",
            "call_to_action": "Learn more about commercial roofing",
        },
        {
            "touchpoint_number": 3,
            "days_after_trigger": 3,
            "time_of_day": "11:00:00",
            "touchpoint_type": "email",
            "subject_line": "How We Helped a Similar {roof_type} Project",
            "content_template": """Hi {first_name},

I wanted to share a recent project similar to yours:

{case_study}

The property manager had concerns about {primary_concern} - just like you mentioned.

After our thorough assessment, we identified the root cause and provided options ranging from targeted repairs to full replacement, with transparent pricing for each approach.

Have you had a chance to connect with our team yet? If not, just reply and we'll reach out at a time that works for you.

Best,
Weathercraft Roofing
""",
            "call_to_action": "Schedule your assessment",
        },
        {
            "touchpoint_number": 4,
            "days_after_trigger": 7,
            "time_of_day": "14:00:00",
            "touchpoint_type": "email",
            "subject_line": "Quick follow-up on your roof assessment, {first_name}",
            "content_template": """Hi {first_name},

I wanted to follow up on your commercial roof assessment request from last week.

Our team has been trying to reach you to schedule your free on-site evaluation. This assessment includes:

✓ Complete membrane and seam inspection
✓ Thermal imaging scan (where applicable)
✓ Drainage analysis
✓ Photo-documented report
✓ Honest repair vs. replacement recommendations

No pressure, no obligation - just a clear picture of your roof's condition.

Would you prefer we call, email, or text to schedule?

Regards,
Weathercraft Roofing
""",
            "call_to_action": "Schedule now",
        },
        {
            "touchpoint_number": 5,
            "days_after_trigger": 14,
            "time_of_day": "10:00:00",
            "touchpoint_type": "email",
            "subject_line": "Last chance: Your complimentary roof assessment",
            "content_template": """Hi {first_name},

This is my final follow-up regarding your commercial roof assessment request.

I understand schedules get busy, but I don't want you to miss out on:

• Free comprehensive inspection (normally $500+ value)
• Priority scheduling this month
• 10% discount on any work if scheduled within 30 days

Your {roof_type} system is {roof_age_years} years old. Regular professional inspections can extend its lifespan significantly and catch small issues before they become expensive problems.

If you're no longer interested, no worries - just let me know and I'll close out your file.

Otherwise, reply "SCHEDULE" and we'll get you on the calendar.

Best regards,
Weathercraft Roofing
""",
            "call_to_action": "Reply SCHEDULE",
        },
    ],
}


def get_roof_type_education(roof_type: str) -> str:
    """Get educational content based on roof type."""
    education = {
        "tpo": """**TPO (Thermoplastic Polyolefin)**
TPO is one of the most popular commercial roofing membranes due to its energy efficiency and durability. Typical lifespan: 20-30 years with proper maintenance. Key vulnerabilities include seam failures, UV degradation, and punctures from foot traffic.""",
        "epdm": """**EPDM (Rubber Roofing)**
EPDM is a proven, cost-effective membrane that's been used for over 50 years. Typical lifespan: 20-25 years. Watch for shrinkage, seam separation, and surface crazing as the membrane ages.""",
        "pvc": """**PVC Membrane**
PVC roofing offers excellent chemical resistance and is ideal for restaurants, manufacturing, and facilities with chemical exposure. Typical lifespan: 25-30 years. Heat-welded seams provide superior waterproofing.""",
        "ssmr": """**Standing Seam Metal Roofing**
Standing seam metal is the premium choice for commercial buildings, offering 40-50+ year lifespans. Key maintenance items include fastener inspection, sealant replacement, and thermal expansion considerations.""",
        "bur": """**Built-Up Roofing (BUR/Tar & Gravel)**
BUR is a traditional multi-layer system that's been used for over 100 years. Typical lifespan: 15-20 years. Common issues include blistering, cracking, and ponding water.""",
        "modified_bitumen": """**Modified Bitumen**
Modified bitumen combines asphalt with rubber or plastic modifiers for improved flexibility. Typical lifespan: 15-20 years. Inspect for membrane splitting, seam failures, and granule loss.""",
        "metal_retrofit": """**Metal Roof Retrofit**
Metal retrofit systems are installed over existing roofing to extend building life. This approach often qualifies for tax incentives and can dramatically improve energy efficiency.""",
        "spray_foam": """**Spray Foam Roofing**
Spray polyurethane foam (SPF) provides excellent insulation and seamless waterproofing. Typical lifespan: 15-20 years with regular recoating. UV coating maintenance is critical.""",
    }
    return education.get(roof_type, "Your commercial roof system requires professional evaluation to assess its current condition and remaining useful life.")


def get_primary_concern_insight(concern: str) -> str:
    """Get insight based on primary concern."""
    insights = {
        "leaks": "Active leaks require immediate attention to prevent interior damage and mold growth. Our inspector will identify the source and recommend the most cost-effective solution.",
        "storm_damage": "Storm damage assessment is time-sensitive for insurance claims. Our documented inspection can support your claim with photographic evidence.",
        "ponding_water": "Ponding water is a leading cause of premature membrane failure. We'll evaluate drainage patterns and recommend solutions.",
        "membrane_failure": "Membrane blistering or failure often indicates trapped moisture or adhesion issues. Early intervention can prevent costly full replacement.",
        "age": "Proactive assessment of aging systems helps you plan and budget for eventual replacement while maximizing remaining useful life.",
        "energy_efficiency": "Commercial roof upgrades can significantly reduce energy costs. We'll evaluate cool roof and insulation options for your building.",
        "insurance_claim": "We work with insurance adjusters regularly and can help document damage to support your claim.",
        "selling_property": "A professional roof assessment adds credibility to property listings and can identify issues before buyer inspections.",
        "maintenance_plan": "Preventive maintenance can extend roof life by 5-10 years. We'll create a customized maintenance schedule for your building.",
    }
    return insights.get(concern, "Our inspector will thoroughly evaluate your roof's condition and provide honest, actionable recommendations.")


def get_case_study(roof_type: str, concern: str) -> str:
    """Get a relevant case study."""
    # Generic commercial case study - can be customized per roof type
    return f"""**Recent {roof_type.upper()} Project - Oklahoma City**

A 45,000 sq ft warehouse owner came to us with similar concerns about {concern.replace('_', ' ')}.

After our assessment, we found that the issue was isolated to 15% of the roof area. Rather than recommending full replacement, we performed targeted repairs and added a maintenance coating system.

Result: The owner saved over $80,000 compared to full replacement, and the roof is now expected to last another 12+ years."""


def enroll_lead_in_commercial_sequence(
    lead_id: str,
    email: str,
    first_name: str,
    roof_type: str,
    square_footage: int | None,
    roof_age_years: int | None,
    primary_concern: str | None,
    preferred_contact_method: str = "phone",
) -> bool:
    """
    Enroll a commercial roof lead in the nurture sequence.

    Called from lead_engine.py when a lead is handed off from MRG.
    """
    if _is_test_email(email):
        logger.info(f"Skipping nurture enrollment for test email: {email}")
        return False

    try:
        conn = _get_db_connection()
        cursor = conn.cursor()

        # Generate personalization data
        personalization = {
            "first_name": first_name or "there",
            "roof_type": roof_type.upper().replace("_", " ") if roof_type else "commercial",
            "square_footage": f"{square_footage:,}" if square_footage else "your",
            "roof_age_years": str(roof_age_years) if roof_age_years else "unknown",
            "primary_concern": (primary_concern or "condition").replace("_", " "),
            "preferred_contact_method": preferred_contact_method,
            "roof_type_education": get_roof_type_education(roof_type or ""),
            "primary_concern_insight": get_primary_concern_insight(primary_concern or ""),
            "case_study": get_case_study(roof_type or "", primary_concern or ""),
        }

        now = datetime.now(timezone.utc)
        enrollment_id = str(uuid.uuid4())

        # Check if already enrolled
        cursor.execute(
            """
            SELECT id FROM ai_lead_enrollments
            WHERE lead_id = %s AND status = 'active'
            LIMIT 1
            """,
            (lead_id,),
        )
        if cursor.fetchone():
            logger.info(f"Lead {lead_id} already enrolled in nurture sequence")
            conn.close()
            return True

        # Create or get sequence ID
        cursor.execute(
            """
            SELECT id FROM ai_nurture_sequences
            WHERE name = %s AND is_active = true
            LIMIT 1
            """,
            (COMMERCIAL_ROOF_SEQUENCE["name"],),
        )
        row = cursor.fetchone()

        if row:
            sequence_id = row[0]
        else:
            # Create the sequence if it doesn't exist
            sequence_id = str(uuid.uuid4())
            cursor.execute(
                """
                INSERT INTO ai_nurture_sequences
                (id, name, sequence_type, target_segment, is_active, configuration)
                VALUES (%s, %s, %s, %s, true, %s)
                """,
                (
                    sequence_id,
                    COMMERCIAL_ROOF_SEQUENCE["name"],
                    COMMERCIAL_ROOF_SEQUENCE["sequence_type"],
                    COMMERCIAL_ROOF_SEQUENCE["target_segment"],
                    Json({"source": "lead_engine", "created_at": now.isoformat()}),
                ),
            )

            # Create touchpoints
            for tp in COMMERCIAL_ROOF_SEQUENCE["touchpoints"]:
                tp_id = str(uuid.uuid4())
                cursor.execute(
                    """
                    INSERT INTO ai_sequence_touchpoints
                    (id, sequence_id, touchpoint_number, touchpoint_type,
                     days_after_trigger, time_of_day, subject_line,
                     content_template, call_to_action)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        tp_id,
                        sequence_id,
                        tp["touchpoint_number"],
                        tp["touchpoint_type"],
                        tp["days_after_trigger"],
                        tp["time_of_day"],
                        tp["subject_line"],
                        tp["content_template"],
                        tp["call_to_action"],
                    ),
                )

        # Create enrollment
        cursor.execute(
            """
            INSERT INTO ai_lead_enrollments
            (id, lead_id, sequence_id, enrollment_date, current_touchpoint, status, metadata)
            VALUES (%s, %s, %s, %s, 1, 'active', %s)
            """,
            (
                enrollment_id,
                lead_id,
                sequence_id,
                now,
                Json({"personalization": personalization, "email": email}),
            ),
        )

        # Schedule emails
        cursor.execute(
            """
            SELECT id, touchpoint_number, days_after_trigger, time_of_day,
                   subject_line, content_template
            FROM ai_sequence_touchpoints
            WHERE sequence_id = %s
            ORDER BY touchpoint_number
            """,
            (sequence_id,),
        )
        touchpoints = cursor.fetchall()

        for tp in touchpoints:
            tp_id, tp_num, days, time_of_day, subject, body = tp

            # Calculate scheduled time
            scheduled_date = now + timedelta(days=days)
            if time_of_day:
                hour, minute, second = map(int, str(time_of_day).split(":"))
                scheduled_date = scheduled_date.replace(hour=hour, minute=minute, second=0)

            # Personalize content
            personalized_subject = subject
            personalized_body = body
            for key, value in personalization.items():
                personalized_subject = personalized_subject.replace(f"{{{key}}}", str(value))
                personalized_body = personalized_body.replace(f"{{{key}}}", str(value))

            # Queue the email
            cursor.execute(
                """
                INSERT INTO ai_email_queue
                (id, to_email, subject, body, scheduled_for, status, metadata)
                VALUES (%s, %s, %s, %s, %s, 'queued', %s)
                """,
                (
                    str(uuid.uuid4()),
                    email,
                    personalized_subject,
                    personalized_body,
                    scheduled_date,
                    Json({
                        "enrollment_id": enrollment_id,
                        "touchpoint_id": tp_id,
                        "touchpoint_number": tp_num,
                        "sequence": "commercial_roof_advisory",
                        "lead_id": lead_id,
                    }),
                ),
            )

        conn.commit()
        conn.close()

        logger.info(
            f"Enrolled lead {lead_id} ({email}) in commercial roof nurture sequence. "
            f"{len(touchpoints)} emails scheduled."
        )
        return True

    except Exception as e:
        logger.error(f"Failed to enroll lead in nurture sequence: {e}", exc_info=True)
        return False


def update_lead_engagement(lead_id: str, event_type: str, event_data: dict = None) -> bool:
    """
    Update lead engagement from feedback (email opens, clicks, responses).

    event_type: 'opened', 'clicked', 'responded', 'scheduled', 'converted', 'opted_out'
    """
    try:
        conn = _get_db_connection()
        cursor = conn.cursor()

        # Update enrollment engagement score
        score_impact = {
            "opened": 0.1,
            "clicked": 0.3,
            "responded": 0.5,
            "scheduled": 1.0,
            "converted": 2.0,
            "opted_out": -1.0,
        }.get(event_type, 0)

        if event_type == "opted_out":
            cursor.execute(
                """
                UPDATE ai_lead_enrollments
                SET status = 'opted_out', opt_out_date = NOW()
                WHERE lead_id = %s AND status = 'active'
                """,
                (lead_id,),
            )
        elif event_type in ("scheduled", "converted"):
            cursor.execute(
                """
                UPDATE ai_lead_enrollments
                SET status = 'completed', completion_date = NOW(),
                    engagement_score = COALESCE(engagement_score, 0) + %s
                WHERE lead_id = %s AND status = 'active'
                """,
                (score_impact, lead_id),
            )
        else:
            cursor.execute(
                """
                UPDATE ai_lead_enrollments
                SET engagement_score = COALESCE(engagement_score, 0) + %s
                WHERE lead_id = %s AND status = 'active'
                """,
                (score_impact, lead_id),
            )

        # Log engagement event
        cursor.execute(
            """
            INSERT INTO ai_nurture_engagement
            (id, enrollment_id, engagement_type, engagement_timestamp, engagement_data)
            SELECT %s, e.id, %s, NOW(), %s
            FROM ai_lead_enrollments e
            WHERE e.lead_id = %s AND e.status IN ('active', 'completed', 'opted_out')
            LIMIT 1
            """,
            (
                str(uuid.uuid4()),
                event_type,
                Json(event_data or {}),
                lead_id,
            ),
        )

        conn.commit()
        conn.close()

        logger.info(f"Updated engagement for lead {lead_id}: {event_type}")
        return True

    except Exception as e:
        logger.error(f"Failed to update lead engagement: {e}", exc_info=True)
        return False
