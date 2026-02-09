"""
Daily Weathercraft Intelligence Report
=======================================
Generates and emails a daily prospect intelligence report to matthew@weathercraft.net.

Pulls from revenue_leads (real prospects only) and formats into sections:
- Active Bids (bid_type=active with deadlines)
- Property Management Targets (prospect_type=property_mgmt)
- Pipeline / Big Projects (prospect_type=pipeline)
- New Prospects (added in last 24h)
- Outreach Status Summary

Scheduled via agent_scheduler every 24 hours.

Author: BrainOps AI System
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

WC_REPORT_RECIPIENT = "matthew@weathercraft.net"
REPORT_FROM = os.getenv("RESEND_FROM_EMAIL", "Matt @ BrainStack <matt@myroofgenius.com>")


async def generate_daily_wc_report() -> dict[str, Any]:
    """
    Generate and email the daily Weathercraft intelligence report.

    Returns stats about the report generation.
    """
    try:
        from database.async_connection import get_pool
        from email_scheduler_daemon import schedule_nurture_email
    except ImportError as e:
        logger.error("Required modules not available: %s", e)
        return {"error": str(e)}

    pool = get_pool()
    if not pool:
        return {"error": "Database pool not available"}

    now = datetime.now(timezone.utc)
    stats = {
        "timestamp": now.isoformat(),
        "active_bids": 0,
        "property_mgmt": 0,
        "pipeline": 0,
        "new_24h": 0,
        "total_prospects": 0,
        "report_sent": False,
    }

    try:
        # Fetch all real prospects
        prospects = await pool.fetch("""
            SELECT id, company_name, contact_name, email, phone, website,
                   source, stage, status, score, location, metadata,
                   created_at, updated_at
            FROM revenue_leads
            WHERE (is_test = false OR is_test IS NULL)
              AND (is_demo = false OR is_demo IS NULL)
            ORDER BY score DESC
        """)

        if not prospects:
            logger.info("No real prospects found - skipping daily report")
            return stats

        stats["total_prospects"] = len(prospects)

        # Categorize prospects
        active_bids = []
        property_mgmt = []
        pipeline = []
        new_24h = []
        other_prospects = []

        for p in prospects:
            meta = _parse_meta(p["metadata"])
            bid_type = meta.get("bid_type", "")
            prospect_type = meta.get("prospect_type", "")

            row = {
                "company": p["company_name"],
                "contact": p["contact_name"] or "",
                "location": p["location"] or "",
                "score": p["score"] or 0,
                "stage": p["stage"] or "new",
                "notes": meta.get("notes", ""),
                "roof_system": meta.get("roof_system", ""),
                "building_type": meta.get("building_type", ""),
                "estimated_sqft": meta.get("estimated_sqft", ""),
                "estimated_value": meta.get("estimated_value", ""),
                "bid_deadline": meta.get("bid_deadline", ""),
                "discovery_source": meta.get("discovery_source", ""),
                "created_at": p["created_at"],
            }

            if bid_type == "active":
                active_bids.append(row)
            elif prospect_type == "property_mgmt":
                property_mgmt.append(row)
            elif prospect_type == "pipeline":
                pipeline.append(row)
            else:
                other_prospects.append(row)

            # Check if new in last 24h
            if p["created_at"] and (now - p["created_at"].replace(tzinfo=timezone.utc)).total_seconds() < 86400:
                new_24h.append(row)

        stats["active_bids"] = len(active_bids)
        stats["property_mgmt"] = len(property_mgmt)
        stats["pipeline"] = len(pipeline)
        stats["new_24h"] = len(new_24h)

        # Check outreach status
        outreach_stats = await pool.fetchrow("""
            SELECT
                COUNT(*) FILTER (WHERE status = 'queued') as queued,
                COUNT(*) FILTER (WHERE status = 'sent') as sent,
                COUNT(*) FILTER (WHERE status = 'cancelled') as cancelled
            FROM ai_email_queue
            WHERE metadata->>'campaign_id' = 'co_commercial_reroof'
        """)

        # Build HTML report
        date_str = now.strftime("%B %d, %Y")
        subject = _build_subject(active_bids, property_mgmt, pipeline, new_24h)
        body = _build_html_report(
            date_str, active_bids, property_mgmt, pipeline,
            new_24h, other_prospects, outreach_stats, stats
        )

        # Queue the email
        email_id = await schedule_nurture_email(
            recipient=WC_REPORT_RECIPIENT,
            subject=subject,
            body=body,
            delay_minutes=0,
            metadata={
                "source": "daily_wc_report",
                "report_type": "intelligence_report",
                "campaign_id": "co_commercial_reroof",
                "generated_at": now.isoformat(),
            },
        )

        if email_id:
            stats["report_sent"] = True
            stats["email_id"] = email_id
            logger.info("Daily WC report queued: %s", email_id)
        else:
            logger.error("Failed to queue daily WC report")

    except Exception as e:
        logger.error("Daily WC report generation failed: %s", e, exc_info=True)
        stats["error"] = str(e)

    return stats


def _parse_meta(metadata) -> dict:
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _build_subject(active_bids, property_mgmt, pipeline, new_24h) -> str:
    parts = []
    if active_bids:
        parts.append(f"{len(active_bids)} Active Bid{'s' if len(active_bids) != 1 else ''}")
    if property_mgmt:
        parts.append(f"{len(property_mgmt)} Property Mgmt Target{'s' if len(property_mgmt) != 1 else ''}")
    if pipeline:
        parts.append(f"{len(pipeline)} Pipeline Project{'s' if len(pipeline) != 1 else ''}")
    if new_24h:
        parts.append(f"{len(new_24h)} New")

    summary = " + ".join(parts) if parts else "Pipeline Update"
    return f"CO Commercial Reroof: {summary}"


def _build_html_report(date_str, active_bids, property_mgmt, pipeline,
                       new_24h, other_prospects, outreach_stats, stats) -> str:
    html = []
    html.append('<div style="font-family:Arial,sans-serif;max-width:700px;color:#333;">')
    html.append(f'<h2 style="color:#1a365d;">Colorado Commercial Reroof - Daily Intelligence Report</h2>')
    html.append(f'<p style="color:#666;">Compiled: {date_str} | Weathercraft Roofing Campaign System</p>')
    html.append('<hr style="border:1px solid #e2e8f0;">')

    # Active Bids
    if active_bids:
        html.append('<h3 style="color:#c53030;">ACTIVE BID OPPORTUNITIES (Act Now)</h3>')
        html.append(_build_table(
            ["Project", "Location", "Scope", "Deadline"],
            [[
                f"<strong>{b['company']}</strong><br>{b['roof_system']}",
                b["location"],
                b["notes"][:200],
                b["bid_deadline"] or "Check status",
            ] for b in active_bids]
        ))

    # Property Management
    if property_mgmt:
        html.append('<h3 style="color:#2b6cb0;">PROPERTY MANAGEMENT TARGETS (Outreach Candidates)</h3>')
        html.append('<p style="color:#666;font-size:13px;">Getting on preferred vendor lists = recurring reroof work.</p>')
        html.append(_build_table(
            ["Company", "Location", "Portfolio"],
            [[
                f"<strong>{p['company']}</strong>",
                p["location"],
                p["notes"][:200],
            ] for p in property_mgmt]
        ))

    # Pipeline
    if pipeline:
        html.append('<h3 style="color:#2f855a;">PIPELINE / BIG PROJECTS</h3>')
        for p in pipeline:
            val = f" - {p['estimated_value']}" if p.get("estimated_value") else ""
            sqft = f", {int(p['estimated_sqft']):,} sqft" if p.get("estimated_sqft") else ""
            html.append(f'<p><strong>{p["company"]}</strong>{val}{sqft}. {p["notes"][:200]}</p>')

    # New in last 24h
    if new_24h:
        html.append(f'<h3 style="color:#d69e2e;">NEW IN LAST 24 HOURS ({len(new_24h)})</h3>')
        for n in new_24h:
            html.append(f'<p><strong>{n["company"]}</strong> - {n["location"]}. Score: {n["score"]}. {n["notes"][:150]}</p>')

    # Other prospects
    if other_prospects:
        html.append(f'<h3 style="color:#718096;">OTHER PROSPECTS ({len(other_prospects)})</h3>')
        html.append(_build_table(
            ["Company", "Location", "Score", "Stage"],
            [[
                f"<strong>{o['company']}</strong>",
                o["location"],
                str(o["score"]),
                o["stage"],
            ] for o in other_prospects[:10]]
        ))
        if len(other_prospects) > 10:
            html.append(f'<p style="color:#999;">...and {len(other_prospects) - 10} more</p>')

    # Summary stats
    html.append('<hr style="border:1px solid #e2e8f0;">')
    html.append(f'<p style="color:#666;font-size:12px;">')
    html.append(f'Total Prospects: {stats["total_prospects"]} | ')
    html.append(f'Active Bids: {stats["active_bids"]} | ')
    html.append(f'Property Mgmt: {stats["property_mgmt"]} | ')
    html.append(f'Pipeline: {stats["pipeline"]}')
    if outreach_stats:
        html.append(f' | Emails Sent: {outreach_stats["sent"] or 0}')
        html.append(f' | Queued: {outreach_stats["queued"] or 0}')
    html.append('</p>')

    # Action items
    if active_bids:
        html.append('<h3 style="color:#c53030;">IMMEDIATE ACTION ITEMS</h3>')
        html.append('<ul>')
        for b in active_bids:
            deadline = f" - Deadline: {b['bid_deadline']}" if b.get("bid_deadline") else ""
            html.append(f'<li><strong>BID:</strong> {b["company"]}{deadline}</li>')
        html.append('</ul>')

    # Bid portal reminder
    html.append('<h3 style="color:#4a5568;">BID PORTAL REGISTRATION</h3>')
    html.append('<ul style="font-size:13px;">')
    html.append('<li>BidNet Direct / RMEPS - bidnetdirect.com/colorado (FREE)</li>')
    html.append('<li>SAM.gov - Required for federal/state bids (FREE)</li>')
    html.append('<li>Colorado OSC - osc.colorado.gov/spco/solicitations (FREE)</li>')
    html.append('<li>City of COS Procurement - coloradosprings.gov/procurement (FREE)</li>')
    html.append('</ul>')

    html.append('<hr style="border:1px solid #e2e8f0;">')
    html.append('<p style="color:#999;font-size:11px;">Generated by BrainOps AI Campaign System | ')
    html.append('All bid deadlines should be independently verified via official portals.</p>')
    html.append('</div>')

    return "\n".join(html)


def _build_table(headers: list[str], rows: list[list[str]]) -> str:
    html = ['<table style="width:100%;border-collapse:collapse;margin:10px 0;">']
    html.append('<tr style="background:#f7fafc;">')
    for h in headers:
        html.append(f'<th style="text-align:left;padding:8px;border:1px solid #e2e8f0;">{h}</th>')
    html.append('</tr>')
    for i, row in enumerate(rows):
        bg = ' style="background:#f7fafc;"' if i % 2 == 1 else ""
        html.append(f"<tr{bg}>")
        for cell in row:
            html.append(f'<td style="padding:8px;border:1px solid #e2e8f0;">{cell}</td>')
        html.append("</tr>")
    html.append("</table>")
    return "\n".join(html)
