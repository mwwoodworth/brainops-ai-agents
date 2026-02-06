"""
Campaign Manager
================
Pluggable campaign system for lead generation outreach.

Campaign #1: Colorado Commercial Reroof (Front Range)
- MRG-branded outreach to large commercial building owners
- Weathercraft revealed only at handoff (email 5)
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
        subject="Free AI Roof Assessment for Your {building_type} in {city}",
        body_html="""<div style="font-family:Arial,sans-serif;max-width:600px;color:#333;">
<p>Hi {contact_name},</p>
<p>I'm reaching out because your {building_type} in {city} may be approaching the point where a professional roof evaluation makes sense.</p>
<p>Colorado's Front Range climate is uniquely harsh on commercial roofs:</p>
<ul>
  <li><strong>Hail season (April - September)</strong> causes more commercial roof damage here than almost anywhere in the country</li>
  <li><strong>UV exposure at 6,000+ feet</strong> accelerates membrane degradation 20-30% faster than sea level</li>
  <li><strong>Freeze-thaw cycling</strong> stresses seams and flashings through 300+ cycles per year</li>
</ul>
<p>We built MyRoofGenius to give commercial building owners a fast, honest picture of their roof's condition using AI-powered analysis.</p>
<p><strong>Here's what we offer at no cost:</strong></p>
<ul>
  <li>AI analysis of your roof photos identifying damage, wear, and risk areas</li>
  <li>Condition scoring with estimated remaining useful life</li>
  <li>Repair vs. replacement cost comparison</li>
  <li>PDF report you can share with your team or insurance carrier</li>
</ul>
<p><a href="{cta_url}" style="display:inline-block;background:#2563eb;color:#fff;padding:12px 24px;border-radius:6px;text-decoration:none;font-weight:bold;">Get Your Free Roof Assessment</a></p>
<p>Or simply reply to this email with a few photos of your roof and we'll send back a detailed analysis.</p>
<p>Best,<br>Matt Woodworth<br><em>MyRoofGenius AI</em></p>
</div>""",
        call_to_action="Get free AI roof assessment",
    ),
    EmailTemplate(
        step=2, delay_days=2,
        subject="What {roof_system} Owners in Colorado Should Know",
        body_html="""<div style="font-family:Arial,sans-serif;max-width:600px;color:#333;">
<p>Hi {contact_name},</p>
<p>Quick follow-up with some information specific to your building.</p>
<p>{roof_system_education}</p>
<p><strong>Why Colorado is different for large commercial roofs:</strong></p>
<ul>
  <li>Buildings over 10,000 sqft face unique drainage challenges during Colorado's intense afternoon thunderstorms</li>
  <li>Thermal expansion and contraction at altitude puts more stress on seams and penetrations</li>
  <li>Hail impacts on large flat roofs often go unnoticed until interior leaks appear months later</li>
</ul>
<p>A proactive assessment now, before hail season, can save tens of thousands in emergency repairs later.</p>
<p><a href="{cta_url}" style="display:inline-block;background:#2563eb;color:#fff;padding:12px 24px;border-radius:6px;text-decoration:none;font-weight:bold;">Upload Roof Photos for Free Analysis</a></p>
<p>Best,<br>Matt Woodworth<br><em>MyRoofGenius AI</em></p>
</div>""",
        call_to_action="Upload photos for analysis",
    ),
    EmailTemplate(
        step=3, delay_days=5,
        subject="How a {city} {building_type} Saved $230K on Their Reroof",
        body_html="""<div style="font-family:Arial,sans-serif;max-width:600px;color:#333;">
<p>Hi {contact_name},</p>
<p>I wanted to share a recent example that might be relevant to your situation.</p>
<p><strong>Case: 42,000 sqft Distribution Center - Colorado Springs</strong></p>
<p>The facility manager noticed a few small leaks after a summer hailstorm and assumed the entire membrane needed replacement. The initial estimate from another contractor was $320,000+.</p>
<p>After an AI-assisted assessment combined with a professional on-site inspection, the actual scope was much more targeted:</p>
<ul>
  <li>Only 18% of the membrane showed hail damage requiring attention</li>
  <li>The remaining 82% was in solid condition with 10+ years of useful life</li>
  <li>Targeted repairs + a protective coating system brought total cost to $87,000</li>
  <li><strong>Savings: over $230,000</strong></li>
</ul>
<p>The takeaway: not every roof issue requires full replacement, and an honest assessment is the first step.</p>
<p><a href="{cta_url}" style="display:inline-block;background:#2563eb;color:#fff;padding:12px 24px;border-radius:6px;text-decoration:none;font-weight:bold;">See What We'd Recommend for Your Building</a></p>
<p>Best,<br>Matt Woodworth<br><em>MyRoofGenius AI</em></p>
</div>""",
        call_to_action="Get your assessment",
    ),
    EmailTemplate(
        step=4, delay_days=9,
        subject="Quick question about your {building_type} roof",
        body_html="""<div style="font-family:Arial,sans-serif;max-width:600px;color:#333;">
<p>Hi {contact_name},</p>
<p>Just a brief follow-up.</p>
<p>Colorado hail season starts in April, and scheduling professional inspections gets backed up quickly once storms begin. If you've been considering having your roof evaluated, now is an ideal time.</p>
<p>Our assessment includes:</p>
<ul>
  <li>Complete membrane and seam condition analysis</li>
  <li>Drainage pattern evaluation</li>
  <li>Flashing and penetration seal inspection</li>
  <li>Photo-documented findings report</li>
  <li>Honest repair vs. replacement recommendations with cost ranges</li>
</ul>
<p>Would you prefer to upload photos for an AI analysis, or schedule a professional on-site evaluation?</p>
<p>Just reply to this email and we'll take it from there.</p>
<p>Best,<br>Matt<br><em>MyRoofGenius AI</em></p>
</div>""",
        call_to_action="Reply to schedule",
    ),
    EmailTemplate(
        step=5, delay_days=14,
        subject="Complimentary Professional Roof Inspection - Limited Availability",
        body_html="""<div style="font-family:Arial,sans-serif;max-width:600px;color:#333;">
<p>Hi {contact_name},</p>
<p>This is my last note regarding your commercial roof assessment.</p>
<p>I wanted to share something I think you'll find valuable: we've partnered with <strong>Weathercraft Roofing</strong>, a full-scope commercial roofing specialist based right here in Colorado Springs, to provide complimentary professional on-site inspections for buildings we identify through our AI assessment platform.</p>
<p><strong>Why Weathercraft:</strong></p>
<ul>
  <li>Specialize exclusively in large commercial reroofs (10,000 - 200,000+ sqft)</li>
  <li>All major membrane systems: TPO, EPDM, PVC, BUR, modified bitumen</li>
  <li>ES-1 certified metal shop with in-house fabrication of coping, trim, and flashing</li>
  <li>Standing seam metal roofing and single skin wall panel installation</li>
  <li>Decades of commercial roofing experience in Colorado</li>
</ul>
<p><strong>For a limited time:</strong></p>
<ul>
  <li>Free comprehensive on-site inspection (normally $500+ value)</li>
  <li>Priority scheduling this month</li>
  <li>10% discount on any work if scheduled within 30 days</li>
</ul>
<p>Your {roof_system} system deserves a professional evaluation before storm season. If you're interested, just reply <strong>"SCHEDULE"</strong> and we'll coordinate a convenient time.</p>
<p>If you're not interested, no problem at all. I'll close out your file.</p>
<p>Best regards,<br>Matt Woodworth<br><em>MyRoofGenius AI + Weathercraft Roofing</em></p>
</div>""",
        call_to_action="Reply SCHEDULE",
    ),
]

_CO_REROOF_CAMPAIGN = CampaignConfig(
    id="co_commercial_reroof",
    name="Colorado Commercial Reroof - Front Range",
    campaign_type="commercial_reroof",
    brand="MyRoofGenius",
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
    cta_url="https://myroofgenius.com/free-roof-analysis",
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
