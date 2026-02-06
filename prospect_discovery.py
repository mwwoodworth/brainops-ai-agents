"""
Prospect Discovery Engine
=========================
Discovers and manages commercial building prospects for campaigns.

Stores qualified prospects into revenue_leads with campaign metadata
for tracking through the outreach pipeline.
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


class ProspectDiscoveryEngine:
    """Discovers and manages prospects for campaign outreach."""

    def __init__(self, pool=None):
        self._pool = pool

    def _get_pool(self):
        if self._pool:
            return self._pool
        try:
            from db import get_pool
            return get_pool()
        except Exception:
            return None

    async def add_prospect(
        self,
        campaign_id: str,
        company_name: str,
        contact_name: Optional[str],
        email: str,
        phone: Optional[str] = None,
        website: Optional[str] = None,
        building_type: Optional[str] = None,
        city: Optional[str] = None,
        state: str = "CO",
        estimated_sqft: Optional[int] = None,
        roof_system: Optional[str] = None,
        discovery_source: str = "manual",
    ) -> dict[str, Any]:
        """Add a single prospect to revenue_leads with campaign metadata."""
        pool = self._get_pool()
        if not pool:
            return {"ok": False, "error": "Database not available"}

        email = email.strip().lower()
        if not self._is_valid_email(email):
            return {"ok": False, "error": f"Invalid email: {email}"}

        # Check for duplicate
        existing = await pool.fetchval(
            "SELECT id FROM revenue_leads WHERE email = $1 LIMIT 1", email
        )
        if existing:
            return {"ok": False, "error": "duplicate", "existing_id": str(existing)}

        lead_id = uuid.uuid4()
        now = datetime.now(timezone.utc)

        metadata = {
            "campaign_id": campaign_id,
            "building_type": building_type or "commercial building",
            "city": city,
            "state": state,
            "discovery_source": discovery_source,
            "discovered_at": now.isoformat(),
        }
        if estimated_sqft:
            metadata["estimated_sqft"] = estimated_sqft
        if roof_system:
            metadata["roof_system"] = roof_system

        score = self._score_prospect(building_type, estimated_sqft, contact_name)
        location = f"{city}, {state}" if city else state

        try:
            await pool.execute("""
                INSERT INTO revenue_leads (
                    id, company_name, contact_name, email, phone, website,
                    stage, status, score, source, industry, location, metadata,
                    created_at, updated_at, is_test, is_demo
                ) VALUES (
                    $1, $2, $3, $4, $5, $6,
                    'new', 'new', $7, 'campaign_discovery', 'commercial_roofing', $8, $9,
                    $10, $10, FALSE, FALSE
                )
            """,
                lead_id, company_name, contact_name, email, phone, website,
                score, location, json.dumps(metadata), now,
            )

            logger.info(f"Prospect added: {company_name} ({email}) for campaign {campaign_id}")
            return {
                "ok": True,
                "lead_id": str(lead_id),
                "company_name": company_name,
                "email": email,
                "score": score,
            }
        except Exception as e:
            logger.error(f"Failed to add prospect: {e}")
            return {"ok": False, "error": str(e)}

    async def add_prospects_batch(
        self,
        campaign_id: str,
        prospects: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Add multiple prospects. Returns summary of results."""
        results = {"total": len(prospects), "added": 0, "duplicates": 0, "errors": 0, "items": []}

        for p in prospects:
            result = await self.add_prospect(
                campaign_id=campaign_id,
                company_name=p.get("company_name", "Unknown"),
                contact_name=p.get("contact_name"),
                email=p.get("email", ""),
                phone=p.get("phone"),
                website=p.get("website"),
                building_type=p.get("building_type"),
                city=p.get("city"),
                state=p.get("state", "CO"),
                estimated_sqft=p.get("estimated_sqft"),
                roof_system=p.get("roof_system"),
                discovery_source=p.get("discovery_source", "batch_import"),
            )
            if result.get("ok"):
                results["added"] += 1
            elif result.get("error") == "duplicate":
                results["duplicates"] += 1
            else:
                results["errors"] += 1
            results["items"].append(result)

        return results

    async def discover_from_website_list(
        self,
        campaign_id: str,
        websites: list[str],
        building_type: Optional[str] = None,
        city: Optional[str] = None,
    ) -> dict[str, Any]:
        """Scrape emails from a list of company websites and add as prospects."""
        results = {"total": len(websites), "found": 0, "added": 0, "no_email": 0, "items": []}

        for url in websites:
            try:
                emails = await self.scrape_emails_from_website(url)
                if not emails:
                    results["no_email"] += 1
                    results["items"].append({"website": url, "status": "no_email_found"})
                    continue

                results["found"] += 1
                email = emails[0]
                company = self._company_from_domain(url)

                add_result = await self.add_prospect(
                    campaign_id=campaign_id,
                    company_name=company,
                    contact_name=None,
                    email=email,
                    website=url,
                    building_type=building_type,
                    city=city,
                    discovery_source="website_scrape",
                )
                if add_result.get("ok"):
                    results["added"] += 1
                results["items"].append({
                    "website": url,
                    "email": email,
                    "status": "added" if add_result.get("ok") else add_result.get("error", "failed"),
                })
            except Exception as e:
                results["items"].append({"website": url, "status": f"error: {e}"})

        return results

    async def scrape_emails_from_website(self, website_url: str) -> list[str]:
        """Scrape contact emails from a company website."""
        if not website_url:
            return []
        if not website_url.startswith(("http://", "https://")):
            website_url = f"https://{website_url}"

        pages_to_check = [
            website_url,
            f"{website_url.rstrip('/')}/contact",
            f"{website_url.rstrip('/')}/contact-us",
            f"{website_url.rstrip('/')}/about",
        ]

        found_emails: set[str] = set()
        domain = self._extract_domain(website_url)

        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            for page in pages_to_check:
                try:
                    resp = await client.get(page, headers={"User-Agent": "Mozilla/5.0"})
                    if resp.status_code == 200:
                        emails = re.findall(
                            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                            resp.text,
                        )
                        for e in emails:
                            e_lower = e.lower()
                            e_domain = e_lower.split("@")[-1]
                            if domain and e_domain != domain and not e_domain.endswith(f".{domain}"):
                                continue
                            if any(prefix in e_lower for prefix in [
                                "noreply", "no-reply", "donotreply", "mailer-daemon",
                                "postmaster", "webmaster", "example.com", "test.",
                            ]):
                                continue
                            found_emails.add(e_lower)
                except Exception:
                    continue

        # Sort: prefer named contacts > role addresses > generic
        def _email_sort_key(e: str) -> int:
            local = e.split("@")[0]
            if any(x in local for x in ["info", "contact", "office", "admin", "sales"]):
                return 1
            return 0

        return sorted(found_emails, key=_email_sort_key)

    async def get_campaign_leads(
        self,
        campaign_id: str,
        stage: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get leads for a specific campaign."""
        pool = self._get_pool()
        if not pool:
            return []

        query = """
            SELECT id, company_name, contact_name, email, phone, website,
                   stage, status, score, location, metadata, created_at
            FROM revenue_leads
            WHERE metadata->>'campaign_id' = $1
              AND is_test = FALSE AND is_demo = FALSE
        """
        params: list[Any] = [campaign_id]

        if stage:
            query += " AND stage = $" + str(len(params) + 1)
            params.append(stage)

        query += f" ORDER BY score DESC, created_at DESC LIMIT {limit}"

        rows = await pool.fetch(query, *params)
        results = []
        for r in rows:
            meta = r.get("metadata")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            results.append({
                "id": str(r["id"]),
                "company_name": r["company_name"],
                "contact_name": r["contact_name"],
                "email": r["email"],
                "phone": r["phone"],
                "website": r["website"],
                "stage": r["stage"],
                "status": r["status"],
                "score": r["score"],
                "location": r["location"],
                "metadata": meta or {},
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            })
        return results

    async def get_campaign_stats(self, campaign_id: str) -> dict[str, Any]:
        """Get stats for a campaign's leads."""
        pool = self._get_pool()
        if not pool:
            return {}

        total = await pool.fetchval(
            "SELECT COUNT(*) FROM revenue_leads WHERE metadata->>'campaign_id' = $1 AND is_test = FALSE",
            campaign_id,
        )

        stage_rows = await pool.fetch("""
            SELECT stage, COUNT(*) as cnt
            FROM revenue_leads
            WHERE metadata->>'campaign_id' = $1 AND is_test = FALSE
            GROUP BY stage
        """, campaign_id)

        emails_queued = await pool.fetchval("""
            SELECT COUNT(*) FROM ai_email_queue
            WHERE metadata->>'campaign_id' = $1
        """, campaign_id)

        emails_sent = await pool.fetchval("""
            SELECT COUNT(*) FROM ai_email_queue
            WHERE metadata->>'campaign_id' = $1 AND status = 'sent'
        """, campaign_id)

        return {
            "campaign_id": campaign_id,
            "total_leads": total or 0,
            "by_stage": {r["stage"]: r["cnt"] for r in stage_rows},
            "emails_queued": emails_queued or 0,
            "emails_sent": emails_sent or 0,
        }

    def _score_prospect(
        self,
        building_type: Optional[str],
        estimated_sqft: Optional[int],
        contact_name: Optional[str],
    ) -> int:
        """Score a prospect 0-100 based on fit."""
        score = 30  # base

        high_value_types = {
            "warehouse", "distribution center", "hospital", "medical complex",
            "data center", "university", "airport hangar", "convention center",
        }
        medium_value_types = {
            "office complex", "industrial park", "school", "government building",
            "manufacturing facility",
        }
        if building_type and building_type.lower() in high_value_types:
            score += 25
        elif building_type and building_type.lower() in medium_value_types:
            score += 15
        else:
            score += 5

        if estimated_sqft:
            if estimated_sqft >= 100000:
                score += 25
            elif estimated_sqft >= 50000:
                score += 20
            elif estimated_sqft >= 20000:
                score += 15
            elif estimated_sqft >= 10000:
                score += 10
            else:
                score += 5

        if contact_name:
            score += 10

        return min(score, 100)

    @staticmethod
    def _is_valid_email(email: str) -> bool:
        return bool(re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email))

    @staticmethod
    def _extract_domain(url: str) -> Optional[str]:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url if "://" in url else f"https://{url}")
            host = parsed.hostname or ""
            if host.startswith("www."):
                host = host[4:]
            return host.lower() if host else None
        except Exception:
            return None

    @staticmethod
    def _company_from_domain(url: str) -> str:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url if "://" in url else f"https://{url}")
            host = parsed.hostname or ""
            if host.startswith("www."):
                host = host[4:]
            name = host.split(".")[0] if host else "Unknown"
            return name.replace("-", " ").replace("_", " ").title()
        except Exception:
            return "Unknown"


# Singleton
_discovery_engine: Optional[ProspectDiscoveryEngine] = None


def get_discovery_engine() -> ProspectDiscoveryEngine:
    global _discovery_engine
    if _discovery_engine is None:
        _discovery_engine = ProspectDiscoveryEngine()
    return _discovery_engine
