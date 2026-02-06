"""
Campaign Manager
================
Pluggable campaign system for lead generation outreach.

Campaign #1: Colorado Commercial Reroof (Front Range)
- Direct Weathercraft-branded outreach to large commercial building owners
- 3-email sequence over 5 days, no giveaways, straight to business
- Reusable architecture for future campaigns (SaaS, digital products, AI tools)
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class GeographyConfig:
    states: list[str]
    cities: list[str]
    metro_areas: list[str] = field(default_factory=list)


@dataclass
class EmailTemplate:
    step: int
    delay_days: int
    subject: str
    body_html: str
    call_to_action: str


@dataclass
class HandoffPartner:
    name: str
    location: str
    capabilities: list[str]
    certifications: list[str]
    experience: str
    phone: str = ""
    website: str = ""


@dataclass
class CampaignConfig:
    id: str
    name: str
    campaign_type: str
    brand: str
    from_name: str
    target_audience: str
    geography: GeographyConfig
    building_types: list[str]
    roof_systems: list[str]
    min_sqft: int
    templates: list[EmailTemplate]
    cta_url: str
    handoff_partner: Optional[HandoffPartner]
    daily_outreach_limit: int
    is_active: bool
    physical_address: str
    unsubscribe_url: str


def _email_footer(brand: str, address: str, unsub_url: str, lead_id: str) -> str:
    return f"""
<div style="font-size:11px;color:#999;margin-top:30px;border-top:1px solid #eee;padding-top:10px;font-family:Arial,sans-serif;">
  <p>{brand} | {address}</p>
  <p><a href="{unsub_url}?lid={lead_id}" style="color:#999;">Unsubscribe</a></p>
</div>"""


_CO_REROOF_TEMPLATES = [
    EmailTemplate(
        step=1, delay_days=0,
        subject="Your {building_type} in {city} - Roof Season Is Coming",
        body_html="""<div style="font-family:Arial,sans-serif;max-width:600px;color:#333;">
<p>Hi {contact_name},</p>
<p>I'm Matt Woodworth with <strong>Weathercraft Roofing</strong> in Colorado Springs. We specialize exclusively in large commercial reroofs along the Front Range.</p>
<p>I'm reaching out because Colorado hail season is right around the corner, and buildings like your {building_type} in {city} take a beating every year:</p>
<ul>
  <li><strong>Hail season (April - September)</strong> drives more commercial roof claims in Colorado than almost anywhere in the country</li>
  <li><strong>UV at 6,000+ feet</strong> degrades membranes 20-30% faster than at sea level</li>
  <li><strong>300+ freeze-thaw cycles per year</strong> stress every seam and flashing</li>
</ul>
<p>If your roof is 10+ years old or you've noticed any leaks, ponding, or membrane wear, it's worth getting eyes on it before storm season hits.</p>
<p><strong>What we do:</strong></p>
<ul>
  <li>Large commercial reroofs: 10,000 - 200,000+ sqft</li>
  <li>All membrane systems: TPO, EPDM, PVC, BUR, modified bitumen</li>
  <li>Standing seam metal roofing with in-house ES-1 certified metal shop</li>
  <li>We fabricate our own coping, trim, and flashing - no middleman</li>
</ul>
<p>If you'd like to discuss your roof situation, just reply to this email or give me a call.</p>
<p>Best,<br>Matt Woodworth<br><em>Weathercraft Roofing</em><br>Colorado Springs, CO</p>
</div>""",
        call_to_action="Reply to discuss",
    ),
    EmailTemplate(
        step=2, delay_days=3,
        subject="Quick question about your {building_type} roof, {contact_name}",
        body_html="""<div style="font-family:Arial,sans-serif;max-width:600px;color:#333;">
<p>Hi {contact_name},</p>
<p>Following up on my earlier note. Just wanted to make sure it landed in your inbox.</p>
<p>{roof_system_education}</p>
<p>We recently worked with a facility manager in Colorado Springs whose 42,000 sqft distribution center had hail damage. They were quoted $320K for a full tear-off by another contractor. After our inspection, the actual scope was far more targeted - we saved them over $230,000 by addressing only the damaged areas.</p>
<p>Not every roof issue requires full replacement. Sometimes a targeted repair or coating system is the right call. We'll tell you straight either way.</p>
<p>If you're thinking about your roof at all before hail season, I'd be happy to talk through your options. Just reply here.</p>
<p>Best,<br>Matt Woodworth<br><em>Weathercraft Roofing</em><br>Colorado Springs, CO</p>
</div>""",
        call_to_action="Reply to discuss",
    ),
    EmailTemplate(
        step=3, delay_days=5,
        subject="Last note - {building_type} roof in {city}",
        body_html="""<div style="font-family:Arial,sans-serif;max-width:600px;color:#333;">
<p>Hi {contact_name},</p>
<p>Last note from me on this. I know you're busy.</p>
<p>If your {building_type} roof needs attention this year, we'd like the opportunity to earn your business. Weathercraft has been doing large commercial reroofs in Colorado for decades, and we're one of the few contractors on the Front Range with an in-house ES-1 certified metal shop.</p>
<p>We're not the cheapest, but we're thorough, we're honest, and we stand behind our work.</p>
<p>If the timing isn't right now, no worries at all. But if you'd like to talk before hail season fills up everyone's schedule, just reply or call anytime.</p>
<p>Best regards,<br>Matt Woodworth<br><em>Weathercraft Roofing</em><br>Colorado Springs, CO</p>
</div>""",
        call_to_action="Reply or call",
    ),
]

_CO_REROOF_CAMPAIGN = CampaignConfig(
    id="co_commercial_reroof",
    name="Colorado Commercial Reroof - Front Range",
    campaign_type="commercial_reroof",
    brand="Weathercraft Roofing",
    from_name="Matt Woodworth",
    target_audience="facility_managers",
    geography=GeographyConfig(
        states=["CO"],
        cities=[
            "Colorado Springs", "Denver", "Pueblo", "Fort Collins",
            "Boulder", "Castle Rock", "Monument", "Woodland Park",
            "Canon City", "Aurora", "Lakewood", "Centennial",
            "Thornton", "Arvada", "Westminster", "Broomfield",
            "Longmont", "Loveland", "Greeley", "Parker",
        ],
        metro_areas=["Front Range", "Colorado Springs Metro", "Denver Metro"],
    ),
    building_types=[
        "warehouse", "distribution center", "big-box retail",
        "hospital", "medical complex", "school", "university",
        "government building", "office complex", "industrial park",
        "church", "HOA common building", "strip mall",
        "manufacturing facility", "data center", "airport hangar",
        "convention center", "grocery store", "self-storage facility",
    ],
    roof_systems=["TPO", "EPDM", "PVC", "BUR", "modified bitumen", "standing seam metal"],
    min_sqft=5000,
    templates=_CO_REROOF_TEMPLATES,
    cta_url="https://www.weathercraftroofingco.com",
    handoff_partner=HandoffPartner(
        name="Weathercraft Roofing",
        location="Colorado Springs, CO",
        capabilities=[
            "Large commercial reroofs (10K - 200K+ sqft)",
            "TPO, EPDM, PVC membrane systems",
            "Built-up roofing (BUR)",
            "Modified bitumen",
            "Standing seam metal roofing",
            "ES-1 certified metal shop - in-house coping, trim, flashing fabrication",
            "Single skin wall panel installation",
        ],
        certifications=["ES-1 Certified Metal Shop", "GAF Master Elite", "Owens Corning Preferred"],
        experience="Decades of commercial roofing experience in Colorado",
        website="https://www.weathercraftroofingco.com",
    ),
    daily_outreach_limit=50,
    is_active=True,
    physical_address="Colorado Springs, CO 80903",
    unsubscribe_url="https://myroofgenius.com/unsubscribe",
)

CAMPAIGNS: dict[str, CampaignConfig] = {
    "co_commercial_reroof": _CO_REROOF_CAMPAIGN,
}

ROOF_SYSTEM_EDUCATION: dict[str, str] = {
    "TPO": (
        "<strong>TPO (Thermoplastic Polyolefin)</strong> is one of the most popular commercial "
        "membrane systems due to its energy efficiency and heat-welded seams. Typical lifespan is "
        "20-30 years, but at Colorado's elevation, UV degradation accelerates significantly. "
        "Key watch items: seam integrity after thermal cycling, membrane brittleness from UV exposure, "
        "and puncture damage from hail."
    ),
    "EPDM": (
        "<strong>EPDM (Rubber Roofing)</strong> has been a commercial roofing workhorse for 50+ years. "
        "Typical lifespan is 20-25 years. In Colorado, the primary concerns are shrinkage from "
        "UV/ozone exposure at altitude, seam separation from freeze-thaw cycling, and surface "
        "crazing that leads to slow leaks."
    ),
    "PVC": (
        "<strong>PVC Membrane</strong> offers excellent chemical resistance and heat-welded seams "
        "for superior waterproofing. Lifespan of 25-30 years. Colorado's intense UV can cause "
        "plasticizer migration over time, making periodic inspection critical for buildings at elevation."
    ),
    "BUR": (
        "<strong>Built-Up Roofing (BUR / Tar & Gravel)</strong> is a traditional multi-layer system "
        "with 15-20 year lifespan. Colorado's freeze-thaw cycling causes blistering and cracking "
        "in the asphalt layers. Hail can dislodge gravel ballast, exposing the membrane beneath "
        "to accelerated UV damage."
    ),
    "modified bitumen": (
        "<strong>Modified Bitumen</strong> combines asphalt with rubber or plastic modifiers for "
        "improved flexibility in temperature extremes. Lifespan of 15-20 years. Colorado's wide "
        "daily temperature swings (40-50 degree shifts are common on the Front Range) stress the "
        "membrane and seams more than in moderate climates."
    ),
    "standing seam metal": (
        "<strong>Standing Seam Metal</strong> is the premium choice for commercial buildings, "
        "offering 40-50+ year lifespans. Key Colorado-specific maintenance includes fastener "
        "inspection (thermal expansion/contraction at altitude is more extreme), sealant "
        "replacement, and ensuring snow guards are properly rated for Colorado snow loads."
    ),
}

_DEFAULT_ROOF_EDUCATION = (
    "Your commercial roof system requires professional evaluation to assess its current "
    "condition and remaining useful life, especially given Colorado's demanding climate."
)


def get_campaign(campaign_id: str) -> Optional[CampaignConfig]:
    return CAMPAIGNS.get(campaign_id)


def list_campaigns(active_only: bool = True) -> list[CampaignConfig]:
    if active_only:
        return [c for c in CAMPAIGNS.values() if c.is_active]
    return list(CAMPAIGNS.values())


def get_campaign_templates(campaign_id: str) -> list[EmailTemplate]:
    campaign = CAMPAIGNS.get(campaign_id)
    if not campaign:
        return []
    return campaign.templates


def personalize_template(
    template: EmailTemplate,
    lead_data: dict[str, Any],
    campaign: CampaignConfig,
) -> tuple[str, str]:
    """Personalize an email template with lead data. Returns (subject, body_html) with footer."""
    contact_name = lead_data.get("contact_name") or "there"
    company_name = lead_data.get("company_name") or ""
    metadata = lead_data.get("metadata") or {}
    if isinstance(metadata, str):
        import json
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}

    building_type = metadata.get("building_type", "commercial building")
    city = metadata.get("city") or lead_data.get("location", "").split(",")[0].strip() or "Colorado"
    roof_system = metadata.get("roof_system", "commercial")
    lead_id = str(lead_data.get("id", ""))

    roof_system_education = ROOF_SYSTEM_EDUCATION.get(roof_system, _DEFAULT_ROOF_EDUCATION)

    variables = {
        "contact_name": contact_name,
        "company_name": company_name,
        "building_type": building_type,
        "city": city,
        "roof_system": roof_system,
        "roof_system_education": roof_system_education,
        "cta_url": campaign.cta_url,
        "lead_id": lead_id,
    }

    subject = template.subject
    body = template.body_html
    for key, value in variables.items():
        subject = subject.replace(f"{{{key}}}", str(value))
        body = body.replace(f"{{{key}}}", str(value))

    body += _email_footer(campaign.brand, campaign.physical_address, campaign.unsubscribe_url, lead_id)
    return subject, body


async def notify_lead_to_partner(
    lead_data: dict[str, Any],
    campaign: CampaignConfig,
    event_type: str = "new_prospect",
) -> bool:
    """
    Send a professional lead notification to the handoff partner (matthew@weathercraft.net).

    Triggered when:
    - A new high-score prospect is discovered (score >= 50)
    - A lead is enrolled in outreach
    - A lead replies or shows interest

    The email contains everything needed to follow up: contact info, building details,
    assessment history, and recommended next steps.
    """
    import os
    import json as _json

    partner_email = os.getenv("WC_PARTNER_EMAIL", "matthew@weathercraft.net")

    metadata = lead_data.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = _json.loads(metadata)
        except Exception:
            metadata = {}

    company = lead_data.get("company_name", "Unknown")
    contact = lead_data.get("contact_name") or "Not available"
    email = lead_data.get("email", "Not available")
    phone = lead_data.get("phone") or "Not available"
    website = lead_data.get("website") or "Not available"
    building_type = metadata.get("building_type", "Commercial building")
    city = metadata.get("city") or lead_data.get("location", "").split(",")[0].strip() or "Colorado"
    state = metadata.get("state", "CO")
    sqft = metadata.get("estimated_sqft")
    roof_system = metadata.get("roof_system", "Unknown")
    score = lead_data.get("score", 0)
    lead_id = str(lead_data.get("id", ""))

    event_labels = {
        "new_prospect": "New Commercial Reroof Prospect",
        "outreach_enrolled": "Outreach Sequence Started",
        "reply_received": "LEAD REPLIED - Action Required",
        "high_interest": "HIGH INTEREST - Priority Follow-Up",
    }
    event_label = event_labels.get(event_type, event_type.replace("_", " ").title())

    subject = f"[WC Lead] {event_label}: {company} - {city}, {state}"

    sqft_display = f"{sqft:,} sqft" if sqft else "Unknown"

    body_html = f"""<div style="font-family:Arial,sans-serif;max-width:700px;color:#333;line-height:1.6;">
<div style="background:#1e3a5f;color:#fff;padding:16px 24px;border-radius:8px 8px 0 0;">
  <h2 style="margin:0;font-size:20px;">{event_label}</h2>
  <p style="margin:4px 0 0;opacity:0.85;font-size:13px;">Campaign: {campaign.name} | Generated by MyRoofGenius AI</p>
</div>

<div style="border:1px solid #e0e0e0;border-top:none;padding:24px;border-radius:0 0 8px 8px;">

<table style="width:100%;border-collapse:collapse;margin-bottom:20px;">
  <tr style="background:#f8f9fa;">
    <td style="padding:10px 14px;font-weight:bold;width:35%;border:1px solid #e0e0e0;">Company</td>
    <td style="padding:10px 14px;border:1px solid #e0e0e0;font-size:16px;"><strong>{company}</strong></td>
  </tr>
  <tr>
    <td style="padding:10px 14px;font-weight:bold;border:1px solid #e0e0e0;">Contact</td>
    <td style="padding:10px 14px;border:1px solid #e0e0e0;">{contact}</td>
  </tr>
  <tr style="background:#f8f9fa;">
    <td style="padding:10px 14px;font-weight:bold;border:1px solid #e0e0e0;">Email</td>
    <td style="padding:10px 14px;border:1px solid #e0e0e0;"><a href="mailto:{email}">{email}</a></td>
  </tr>
  <tr>
    <td style="padding:10px 14px;font-weight:bold;border:1px solid #e0e0e0;">Phone</td>
    <td style="padding:10px 14px;border:1px solid #e0e0e0;">{phone}</td>
  </tr>
  <tr style="background:#f8f9fa;">
    <td style="padding:10px 14px;font-weight:bold;border:1px solid #e0e0e0;">Website</td>
    <td style="padding:10px 14px;border:1px solid #e0e0e0;"><a href="{website}">{website}</a></td>
  </tr>
</table>

<h3 style="color:#1e3a5f;margin-top:24px;border-bottom:2px solid #1e3a5f;padding-bottom:6px;">Building Details</h3>
<table style="width:100%;border-collapse:collapse;margin-bottom:20px;">
  <tr style="background:#f8f9fa;">
    <td style="padding:10px 14px;font-weight:bold;width:35%;border:1px solid #e0e0e0;">Building Type</td>
    <td style="padding:10px 14px;border:1px solid #e0e0e0;">{building_type.title()}</td>
  </tr>
  <tr>
    <td style="padding:10px 14px;font-weight:bold;border:1px solid #e0e0e0;">Location</td>
    <td style="padding:10px 14px;border:1px solid #e0e0e0;">{city}, {state}</td>
  </tr>
  <tr style="background:#f8f9fa;">
    <td style="padding:10px 14px;font-weight:bold;border:1px solid #e0e0e0;">Estimated Size</td>
    <td style="padding:10px 14px;border:1px solid #e0e0e0;">{sqft_display}</td>
  </tr>
  <tr>
    <td style="padding:10px 14px;font-weight:bold;border:1px solid #e0e0e0;">Roof System</td>
    <td style="padding:10px 14px;border:1px solid #e0e0e0;">{roof_system}</td>
  </tr>
  <tr style="background:#f8f9fa;">
    <td style="padding:10px 14px;font-weight:bold;border:1px solid #e0e0e0;">Lead Score</td>
    <td style="padding:10px 14px;border:1px solid #e0e0e0;"><strong>{score}/100</strong></td>
  </tr>
</table>

<h3 style="color:#1e3a5f;margin-top:24px;border-bottom:2px solid #1e3a5f;padding-bottom:6px;">Recommended Action</h3>
<div style="background:#e8f5e9;border-left:4px solid #2e7d32;padding:12px 16px;margin-bottom:16px;">
  <p style="margin:0;"><strong>Next step:</strong> Review this prospect and prepare for potential follow-up.
  MRG outreach emails are being sent on your behalf. When the prospect replies to the MRG
  email sequence, you'll receive another notification with their response.</p>
</div>

<p style="font-size:12px;color:#999;margin-top:24px;border-top:1px solid #eee;padding-top:10px;">
  Lead ID: {lead_id}<br>
  This notification was generated by the BrainOps Campaign System.<br>
  Outreach is branded MyRoofGenius. Weathercraft is revealed at handoff (email 5 of 5).
</p>
</div>
</div>"""

    try:
        from database.async_connection import get_pool
        pool = get_pool()
        if pool:
            await pool.execute("""
                INSERT INTO ai_email_queue (id, recipient, subject, body, scheduled_for, status, metadata)
                VALUES ($1, $2, $3, $4, NOW(), 'queued', $5)
            """,
                uuid.uuid4(),
                partner_email,
                subject,
                body_html,
                _json.dumps({
                    "type": "partner_notification",
                    "campaign_id": campaign.id,
                    "lead_id": lead_id,
                    "event_type": event_type,
                }),
            )
            logger.info(f"Partner notification queued for {partner_email}: {event_label} - {company}")
            return True
    except Exception as e:
        logger.error(f"Failed to queue partner notification: {e}")
    return False


def campaign_to_dict(campaign: CampaignConfig) -> dict[str, Any]:
    """Serialize campaign for API responses."""
    return {
        "id": campaign.id,
        "name": campaign.name,
        "campaign_type": campaign.campaign_type,
        "brand": campaign.brand,
        "target_audience": campaign.target_audience,
        "geography": {
            "states": campaign.geography.states,
            "cities": campaign.geography.cities,
            "metro_areas": campaign.geography.metro_areas,
        },
        "building_types": campaign.building_types,
        "roof_systems": campaign.roof_systems,
        "min_sqft": campaign.min_sqft,
        "template_count": len(campaign.templates),
        "cta_url": campaign.cta_url,
        "handoff_partner": campaign.handoff_partner.name if campaign.handoff_partner else None,
        "daily_outreach_limit": campaign.daily_outreach_limit,
        "is_active": campaign.is_active,
    }
