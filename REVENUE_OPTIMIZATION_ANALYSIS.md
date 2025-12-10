# BrainOps Revenue Optimization Analysis

## Data Snapshot
- 9,844 customers, 18,465 jobs, 148 tenants (~66 customers/tenant; ~1.9 jobs/customer; ~125 jobs/tenant)
- Revenue tracking: `invoices`, `payments`, `subscriptions`; pricing + revenue systems already in code (`pricing_quotes`, `pricing_ab_tests`, `revenue_leads`, `revenue_opportunities`, `ai_customer_health`, `ai_churn_predictions`)
- Products: Weathercraft ERP (human-in-loop B2B SaaS for roofers), MyRoofGenius (autonomous B2C/B2B platform), BrainOps AI Agents (backend AI services)

## Immediate Revenue Wins (30-Day)
- Recover unbilled work: find jobs without invoices and bill through InvoicingAgent. Example query: `SELECT tenant_id, COUNT(*) FROM jobs j LEFT JOIN invoices i ON i.job_id = j.id WHERE i.id IS NULL GROUP BY tenant_id ORDER BY COUNT(*) DESC;`
- Convert pay-as-you-go to subscriptions: identify customers with `payments` but no active `subscriptions` and auto-offer monthly/annual plans; prioritize top 15 tenants by job count.
- Reactivate inactive accounts: for customers with no jobs in 180 days or unpaid invoices, trigger Customer Success Agent outreach (discount/concierge) and track in `ai_customer_interventions`.
- Turn on AI pricing A/B tests: use `pricing_ab_tests` to trial +10% base price vs. current and a per-job overage; pick winner by win-rate and margin.
- Bundle Weathercraft + MyRoofGenius: offer "Autonomous Ops" add-on to high-volume Weathercraft tenants (>$X jobs/month) with a time-bound discount; route via Revenue Automation email sequences.

## Untapped Revenue Opportunities
- **Job-to-invoice leakage:** With 18,465 jobs, even small leakage materially impacts ARR. Build a daily guardrail: jobs without invoices <24h old → auto-generate drafts; unpaid >14 days → PaymentAgent nudges.
- **Idle tenants:** Tenants with <5 jobs in last 90 days need activation sequences (training, fast-start templates, pre-built estimates) instead of discounts.
- **Subscription gaps:** Customers with recent payments but `subscriptions.status != 'active'` → auto-start subscription and backfill proration; require human approval only for disputed charges.
- **Usage expansion:** Average ~1.9 jobs/customer implies low workflow penetration. Push "full-lifecycle" bundles (estimate → schedule → invoice) and measure lift in jobs/customer.
- **MyRoofGenius monetization:** Use existing Stripe links in `revenue_automation.py` to sell roof inspection + maintenance plans to homeowners captured by MyRoofGenius flows; upsell contractor leads to ERP seats.

## Cross-Sell / Upsell
- **Weathercraft → MyRoofGenius autonomy:** Target top decile tenants by jobs for AI-autonomous scheduling/lead-gen add-on; price as +per-job or +per-crew package.
- **Homeowner maintenance plans:** Convert B2C leads to recurring maintenance/inspection subscriptions; include "emergency response" upsell for storm season.
- **ERP advanced modules:** Offer AI lead scoring, dynamic pricing, and logistics optimizers as premium modules to tenants with on-time payments and >3 jobs/customer.
- **Annual prepay:** Present annual plans with 10–15% incentive to tenants with DSO <15 days; improves cash and reduces churn.

## Pricing Optimization
- **Segmented tiers:** Use `customer_segment` and volume signals to map to strategies: penetration for low-volume (<10 jobs/mo), value-based for mid/high, dynamic for enterprise/multi-crew.
- **Per-job + floor:** Keep a base platform fee + per-job overage after a bundled allowance (e.g., first 50 jobs included) to capture heavy users while staying SMB-friendly.
- **A/B harness:** Run `pricing_ab_tests` on (a) +10% base, (b) +per-job overage, (c) bundle discount for ERP+autonomy; measure win probability and margin in `pricing_history`.
- **Success-based add-on:** For MyRoofGenius, trial a performance fee on booked jobs/leads while keeping a smaller platform base to reduce price sensitivity.
- **Localization:** Use market/tenant region to adjust for demand spikes (storm seasons) via the dynamic pricing strategy; precompute surge bands to avoid surprising customers.

## Churn Risk Factors & Playbook
- Signals already in code: inactivity >180 days, unpaid invoices, low job count (<2), short relationship, and sentiment drop. Push daily scoring into `ai_customer_health`/`ai_churn_predictions`.
- Intervention rules: unpaid invoices → offer partial payment/plan; inactivity → book training + seed 3 templates; low adoption → enable workflows that auto-create jobs from accepted estimates.
- Data-quality fix: ensure job–customer linkage completeness (previous audit showed gaps) so churn and LTV are accurate; block outreach if linkage <95% to avoid noisy messaging.
- Monitor DSO: rising days-to-pay is an early churn/credit risk indicator; alert when DSO moves >7 days month-over-month.

## High-Value Customer Patterns to Amplify
- Tenants with high jobs/customer (>3) and fast pay (DSO <15) are ideal for premium bundles and annual prepay.
- Customers with recurring jobs (maintenance/inspection cycles) show higher LTV; auto-enroll them into scheduled service plans.
- Multi-location contractors (high customer counts per tenant) justify enterprise onboarding, custom SLA, and per-crew pricing.
- Positive payment history and no disputes correlate with upsell success—prioritize these for new AI modules (dynamic pricing, logistics optimizer).

## Operational Next Steps
- Turn on the pricing engine and revenue automation cron; log results to `pricing_history` and `revenue_metrics`.
- Add two daily monitors: (1) unbilled jobs, (2) churn-risk customers; pipe alerts to RevOps with tenant/customer lists.
- Launch a 2-week A/B on ERP base price vs. per-job overage; choose winner by conversion + margin, then roll to all new tenants.
- Build a cross-sell playbook for the top 15 tenants by job volume; set target attach rate and track in `revenue_opportunities`.
