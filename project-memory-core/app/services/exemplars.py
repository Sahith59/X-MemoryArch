"""
Sub-phases 1.20 + 1.40 — Exemplar bank for semantic classification.

Sub-phase 1.40 changes:
  TECHNICAL_EXEMPLARS renamed to SUBSTANTIVE_EXEMPLARS, expanded to 30 sentences
  covering all domain categories so the gate accepts design, research, business,
  marketing, and personal work content — not just software engineering.

  TYPE_EXEMPLARS restructured into a domain-stratified EXEMPLARS dict-of-dicts.
  The semantic classifier computes per-domain-group centroids from this structure,
  enabling domain-weighted type classification.

  DOMAIN_GROUP maps all 16 project domains to 6 canonical exemplar groups:
  software, design, research, business, marketing, general.

  Backward-compat TYPE_EXEMPLARS (flat list per type) is computed at module load
  by flattening all domain examples. Existing code that imports TYPE_EXEMPLARS
  continues to work unchanged.
"""

# ---------------------------------------------------------------------------
# Gate exemplars — "is this substantive work content vs casual filler?"
# ---------------------------------------------------------------------------

SUBSTANTIVE_EXEMPLARS: list[str] = [
    # Software / technical (12)
    "We decided to use PostgreSQL as our primary database for better JSON support.",
    "Authentication throws a TypeError when the session token is null or has expired.",
    "The API endpoint returns a 500 error when the required header is missing.",
    "We chose FastAPI over Flask due to its built-in type validation and OpenAPI docs.",
    "The database query causes N+1 issues on relationship joins in the ORM layer.",
    "The migration script adds a composite index on project_id and created_at.",
    "The application uses a layered architecture: router, service, and repository layers.",
    "The system must not store personally identifiable information outside encrypted storage.",
    "Run alembic upgrade head to apply all pending database migrations before starting.",
    "We decided to use Redis for session caching because it provides atomic low-latency reads.",
    "We decided on a monorepo structure because it simplifies shared dependency management.",
    "We decided to use Docker for containerizing all services to ensure consistent environments.",
    # Design / creative (5)
    "The design system uses an 8-column grid with 16-pixel gutters across all breakpoints.",
    "Typography follows three scales: display at 48px, body at 16px, and caption at 12px.",
    "The color palette was updated to improve contrast ratios for WCAG AA compliance.",
    "The hero section does not render correctly on viewport widths below 375 pixels.",
    "We chose Figma over Sketch because of its real-time multiplayer collaboration.",
    # Research / education (5)
    "The study found a significant correlation between sleep quality and memory retention.",
    "Participants in the control group showed no measurable change in test scores.",
    "We selected a qualitative approach because the sample size was insufficient for statistics.",
    "The hypothesis was not supported by the data collected in the second testing round.",
    "Statistical significance was achieved at p less than 0.05 in the primary outcome.",
    # Business / product / professional (5)
    "Customer churn increased 15 percent after the pricing change was implemented last quarter.",
    "We pivoted to a freemium model to improve the top-of-funnel acquisition rate.",
    "Revenue declined 12 percent following the introduction of the new pricing tiers in Q2.",
    "The approval bottleneck is delaying new client contracts by an average of two weeks.",
    "Contract renewal requires legal review of three modified SLA clauses before signing.",
    # Marketing / sales (4)
    "Email open rates dropped 40 percent after migrating to the new sending domain.",
    "Paid cost per acquisition is now three times higher than organic due to ad competition.",
    "The landing page conversion rate improved from two to eight percent after the redesign.",
    "Email deliverability dropped after the domain change and is blocking the Q2 campaign.",
    # Personal / general work — specific outcomes, not offers or follow-up phrases (7)
    "The Q3 retrospective revealed three recurring deployment blockers in the release process.",
    "The project timeline shifted by two weeks after scope expansion added three new features.",
    "Focus sessions increased from two to four hours daily after switching to time-blocking.",
    "The proposal deadline is Friday and two sections still require stakeholder approval.",
    "The quarterly budget review revealed a 20 percent overspend on infrastructure costs.",
    "The budget spreadsheet review must be completed before Thursday to finalize the Q2 cost report.",
    "Weekly task list review on Thursday surfaces three overdue items requiring immediate action.",
]

# ---------------------------------------------------------------------------
# Domain group mapping — 16 project domains → 6 canonical exemplar groups
# ---------------------------------------------------------------------------

DOMAIN_GROUP: dict[str, str] = {
    "software":  "software",
    "data":      "software",
    "security":  "software",
    "design":    "design",
    "creative":  "design",
    "research":  "research",
    "education": "research",
    "product":   "business",
    "business":  "business",
    "finance":   "business",
    "legal":     "business",
    "hr":        "business",
    "marketing": "marketing",
    "sales":     "marketing",
    "personal":  "general",
    "general":   "general",
}

# ---------------------------------------------------------------------------
# Domain-stratified type exemplars
# 12 types × 6 domain groups × 2–5 exemplars per combination
# ---------------------------------------------------------------------------

EXEMPLARS: dict[str, dict[str, list[str]]] = {

    # ── decision ──────────────────────────────────────────────────────────────
    "decision": {
        "software": [
            "We decided to use PostgreSQL instead of MySQL for better JSON support.",
            "After evaluating options, we chose FastAPI as our web framework.",
            "We are going with Redux for state management because of its debugging tools.",
            "We decided to defer authentication to Phase 2 to keep scope manageable.",
            "We chose Alembic as the migration tool because of its deep SQLAlchemy integration.",
            "We decided to use FastAPI as the primary web framework for this project.",
            "We decided to use Redis for session caching to reduce database load.",
            "We chose TypeScript over JavaScript to get static type checking across the codebase.",
            "We decided to go with Docker for containerizing all services in the stack.",
        ],
        "design": [
            "We chose Figma over Sketch because of its better real-time collaboration.",
            "We decided on a card-based layout to improve content scannability.",
            "After testing, we agreed to use an 8-column grid as the baseline layout system.",
        ],
        "research": [
            "We selected a qualitative approach because the sample size was too small.",
            "We decided to exclude outliers beyond two standard deviations from the analysis.",
            "We chose a longitudinal study design to capture behavioral changes over time.",
        ],
        "business": [
            "We pivoted to a freemium model to improve top-of-funnel conversion rates.",
            "We decided to enter the SMB segment before enterprise to validate pricing.",
            "The leadership team decided to deprioritize international expansion until Q4.",
        ],
        "marketing": [
            "We decided to prioritize SEO over paid acquisition for sustainable growth.",
            "We chose email re-engagement over push notifications for the retention campaign.",
            "After testing, we decided to sunset the podcast channel due to low ROI.",
        ],
        "general": [
            "We decided to change the approach because the previous one was not working.",
            "After discussion, we agreed to delay the launch until the quality bar is met.",
            "The team decided to simplify the process to improve stakeholder adoption.",
        ],
    },

    # ── problem ───────────────────────────────────────────────────────────────
    "problem": {
        "software": [
            "The login endpoint throws a 500 error when the email field is missing.",
            "TypeError: cannot read property id of undefined in the session handler.",
            "The migration crashes with a foreign key constraint violation on deploy.",
            "Race condition in session creation causes duplicate records occasionally.",
            "CORS error blocks the frontend from reaching the API in production.",
        ],
        "design": [
            "The CTA button is invisible to low-vision users due to insufficient contrast.",
            "Mobile layout breaks at 375px because the grid does not collapse correctly.",
            "The onboarding modal overlaps the navigation bar on small screens.",
        ],
        "research": [
            "Participants consistently misunderstood question 4 due to ambiguous phrasing.",
            "The sample is not representative because all participants were under 30.",
            "Data collection was inconsistent because the survey tool timed out on mobile.",
        ],
        "business": [
            "Customer churn increased 15 percent after the pricing change last quarter.",
            "Onboarding drop-off is 70 percent at step 2 due to confusing instructions.",
            "The approval bottleneck is delaying all new client contracts by two weeks.",
        ],
        "marketing": [
            "Email open rates dropped 40 percent after the domain migration.",
            "Email deliverability is our main blocker for growth this quarter.",
            "Ad spend efficiency dropped after the privacy changes limited tracking.",
        ],
        "general": [
            "The process is not working as expected and is causing delays for the team.",
            "There is a critical issue blocking progress on the main objective.",
            "The current approach is failing and needs to be changed urgently.",
        ],
    },

    # ── structure ─────────────────────────────────────────────────────────────
    "structure": {
        "software": [
            "The system uses a layered architecture: router, service, repository, database.",
            "We designed the authentication flow using a stateless JWT approach.",
            "We separated read and write operations using the CQRS pattern for scalability.",
            "The service layer handles all business rules and routers only do HTTP translation.",
        ],
        "design": [
            "The design system uses an 8-point grid with four breakpoints for layout.",
            "Information architecture groups content into three top-level categories.",
            "We use atomic design: atoms, molecules, organisms, templates, and pages.",
            "The navigation hierarchy has three levels: global, section, and contextual.",
        ],
        "research": [
            "The study uses a mixed-methods framework combining surveys and qualitative interviews.",
            "The analysis framework follows inductive coding in three iterative passes.",
            "The research is structured into four phases: discovery, synthesis, testing, reporting.",
        ],
        "business": [
            "The team is organized into squads aligned to customer journey stages.",
            "The product is structured around three user roles: admin, editor, and viewer.",
            "The content hierarchy places conversion goals above brand storytelling.",
        ],
        "marketing": [
            "The campaign structure uses three funnel stages: awareness, consideration, conversion.",
            "Email sequences are organized by lifecycle stage: onboarding, activation, retention.",
        ],
        "general": [
            "The overall structure follows a three-tier approach: input, process, output.",
            "The workflow is organized into sequential phases with clear handoff criteria.",
        ],
    },

    # ── how_to ────────────────────────────────────────────────────────────────
    "how_to": {
        "software": [
            "Run pip install requirements.txt to install all project dependencies.",
            "Run alembic upgrade head to apply all pending database migrations.",
            "Use uvicorn app.main:app with reload flag to start the development server.",
            "Install Docker Desktop and run docker-compose up to start local services.",
            "Run pytest with the -v flag to see detailed output for each test case.",
            "Run the linter with ruff check . before opening a pull request.",
            "Execute the seed script with python seed.py to populate the local database.",
        ],
        "design": [
            "To run a design sprint: align on the problem, sketch, decide, prototype, test.",
            "To handoff designs: export assets at 1x and 2x, annotate interactions, share the link.",
            "To conduct a usability test: recruit five users, prepare tasks, record sessions.",
        ],
        "research": [
            "To conduct a user interview: prepare a guide, record with consent, transcribe within 24 hours.",
            "To run an A/B test: define hypothesis, set sample size, run for two weeks, analyze.",
            "To code qualitative data: define categories, apply codes, check inter-rater reliability.",
        ],
        "business": [
            "To onboard a new client: send welcome email, schedule kick-off call, share credentials.",
            "To run a quarterly review: collect metrics, compare to targets, identify blockers.",
            "To create a project brief: define scope, budget, timeline, and success criteria.",
        ],
        "marketing": [
            "To launch an email campaign: write copy, design template, set up sequence, A/B test.",
            "To run a paid campaign: define audience, set budget, create ad variations, track conversions.",
        ],
        "general": [
            "To set up the project: clone the repo, copy the env file, install dependencies, run migrations.",
            "To prepare for the meeting: review the agenda, gather data, prepare talking points.",
        ],
    },

    # ── constraint ────────────────────────────────────────────────────────────
    "constraint": {
        "software": [
            "The system must not store any personally identifiable information locally.",
            "All API responses must complete within 200 milliseconds for the search endpoint.",
            "We cannot use external APIs in Phase 1 and everything must work fully offline.",
            "The embedding dimension must be fixed at 384 to match the deployed model.",
        ],
        "design": [
            "The design must meet WCAG AA accessibility standards for all interactive elements.",
            "All components must render correctly at viewport widths between 320 and 1440 pixels.",
            "Brand colors and typography must not be modified without approval from the brand team.",
        ],
        "research": [
            "All participant data must be anonymized before analysis per IRB protocol.",
            "The study must maintain a minimum of 30 participants per condition for statistical validity.",
            "Survey instruments must be validated before use in primary data collection.",
        ],
        "business": [
            "All contracts must be reviewed by legal before signing regardless of deal size.",
            "Budget changes above 10 percent require executive approval before proceeding.",
            "Customer data must comply with GDPR and cannot be stored outside the EU region.",
        ],
        "marketing": [
            "All promotional materials must be reviewed by legal before public release.",
            "Email campaigns must include an unsubscribe option per CAN-SPAM compliance.",
        ],
        "general": [
            "All decisions above a certain threshold require sign-off from the team lead.",
            "Deadlines agreed with external stakeholders cannot be changed without notification.",
        ],
    },

    # ── open_question ─────────────────────────────────────────────────────────
    "open_question": {
        "software": [
            "Should we use cursor-based pagination or offset-based for the memories API?",
            "How should we handle session content that exceeds the context window limit?",
            "Should extraction run synchronously or be queued as a background task?",
        ],
        "design": [
            "Should we use a tab navigation or a hamburger menu for the mobile experience?",
            "How do we handle the information density difference between desktop and mobile?",
            "Should the onboarding flow be linear or allow users to skip ahead?",
        ],
        "research": [
            "Should we use quantitative or qualitative methods for the second study phase?",
            "How do we account for selection bias in the participant recruitment process?",
            "What is the right sample size to achieve statistical power for this analysis?",
        ],
        "business": [
            "Should we expand to enterprise before fully validating SMB product-market fit?",
            "How do we balance growth speed against quality of the customer experience?",
            "Should we build the feature internally or license an existing third-party solution?",
        ],
        "marketing": [
            "Should we invest in content marketing or paid acquisition for this growth phase?",
            "How do we attribute conversions accurately across multiple marketing touchpoints?",
        ],
        "general": [
            "What is the right approach to solve this problem given current constraints?",
            "How do we balance the competing priorities to reach the best overall outcome?",
        ],
    },

    # ── insight ───────────────────────────────────────────────────────────────
    "insight": {
        "software": [
            "SQLite FTS5 performs comparably to Elasticsearch for datasets under 100k records.",
            "Memory deduplication reduces extraction noise by about 30 percent on repeated sessions.",
            "Context windows of three sentences capture decision reasoning much better than one.",
            "Storing embeddings at write time eliminates re-embedding latency at retrieval time.",
        ],
        "design": [
            "Users spend 80 percent of their time on three core screens in session analytics.",
            "Reducing onboarding from 7 steps to 4 increased activation rate by 25 percent.",
            "High contrast alone is insufficient — icon labels are essential for low-vision users.",
            "Card-based layouts outperform list views for exploratory content discovery.",
        ],
        "research": [
            "The study found a significant correlation between sleep quality and retention outcomes.",
            "Participants who received structured feedback performed 40 percent better on follow-up.",
            "The qualitative data revealed behavioral patterns not visible in the survey results.",
            "Replication of the study confirmed the effect size within expected variance.",
        ],
        "business": [
            "Reducing time-to-value in onboarding from three days to same-day doubled 30-day retention.",
            "The top 20 percent of customers generate 80 percent of recurring revenue.",
            "Pricing anchoring with a premium tier increased mid-tier plan selection by 35 percent.",
        ],
        "marketing": [
            "Personalized subject lines increase open rates by 20 percent versus generic ones.",
            "Organic search converts three times better than paid at the same intent level.",
            "Retention campaigns targeting day-7 churn signals doubled 30-day retention.",
        ],
        "general": [
            "Simplifying the process by removing two steps reduced completion time by half.",
            "Regular structured reviews prevent small issues from becoming critical blockers.",
            "Early stakeholder involvement significantly reduces rework at the implementation stage.",
        ],
    },

    # ── workflow_pattern ──────────────────────────────────────────────────────
    "workflow_pattern": {
        "software": [
            "Our deployment process: run tests, build Docker image, push to registry, apply migrations.",
            "For every new feature: write the failing test first, then implement code to pass it.",
            "Bug fix process: reproduce with a failing test, fix the code, verify the test passes.",
            "Database schema changes always go through a versioned migration system, never raw SQL.",
        ],
        "design": [
            "Design review cycle: designer presents, team critiques, revisions made, stakeholder signs off.",
            "Component creation workflow: sketch, prototype in Figma, test with users, finalize tokens.",
            "Usability testing cadence: recruit weekly, run two sessions, synthesize, iterate.",
        ],
        "research": [
            "Research delivery cycle: kick-off, fieldwork, synthesis, report, share-out, archive.",
            "Interview analysis workflow: transcribe, code, theme, validate, present findings.",
            "Literature review process: search, filter, extract, synthesize, cite consistently.",
        ],
        "business": [
            "Client delivery workflow: proposal, kick-off, weekly check-ins, milestone reviews, sign-off.",
            "Quarterly planning: review KPIs, set OKRs, allocate budget, assign owners, track progress.",
        ],
        "marketing": [
            "Campaign launch checklist: brief, copy, design, legal review, schedule, monitor, report.",
            "Content publishing workflow: ideate, write, edit, design, schedule, distribute, measure.",
        ],
        "general": [
            "The weekly review: capture what happened, identify blockers, plan next actions.",
            "Decision-making process: gather context, list options, evaluate trade-offs, decide, document.",
        ],
    },

    # ── task ──────────────────────────────────────────────────────────────────
    "task": {
        "software": [
            "TODO: add input validation for the session date field in the sessions router.",
            "We need to implement rate limiting on the authentication endpoints as a priority.",
            "Fix the pagination bug where offset larger than total count returns a server error.",
            "Set up a GitHub Actions workflow for automated test runs on every pull request.",
        ],
        "design": [
            "TODO: update the button component to meet WCAG AA contrast requirements.",
            "We need to redesign the empty state screens before the next sprint review.",
            "Create high-fidelity mockups for the mobile checkout flow by end of week.",
        ],
        "research": [
            "TODO: recruit 10 additional participants for the second study cohort.",
            "We need to analyze the remaining 15 interview transcripts before the report deadline.",
            "Schedule follow-up interviews with participants who scored below the median.",
        ],
        "business": [
            "TODO: draft the Q3 budget proposal before the finance review on Thursday.",
            "We need to update the client onboarding checklist based on recent feedback.",
            "I need to review the contract terms before the renewal meeting next week.",
        ],
        "marketing": [
            "TODO: update the email templates with the new brand colors before next campaign.",
            "We need to audit the paid ad campaigns before the monthly budget review.",
            "Prepare the monthly marketing report for the leadership review on Friday.",
        ],
        "general": [
            "I need to follow up with the client about the proposal before end of day.",
            "We need to prepare the presentation before the stakeholder meeting on Thursday.",
            "TODO: review the quarterly report and submit it to the finance team by Friday.",
        ],
    },

    # ── reference_context ─────────────────────────────────────────────────────
    "reference_context": {
        "software": [
            "The create_memory function in crud.py handles quality score computation on insert.",
            "The semantic_classifier module is imported lazily so server startup stays fast.",
            "The embedding column stores float32 bytes from the all-MiniLM-L6-v2 model.",
            "The memory_service extraction pipeline lives in app/services/ directory.",
        ],
        "design": [
            "The Figma file at /brand/colors contains all approved color tokens and usage rules.",
            "The hero component in the design system requires a minimum width of 320px.",
            "Component documentation is in the Storybook instance linked from the project README.",
        ],
        "research": [
            "The survey data is stored in data/raw/survey_2026.csv with participant IDs anonymized.",
            "The literature review spreadsheet covers 47 papers from 2020 to 2026.",
            "The interview recordings are in the shared drive under Research/Fieldwork/2026.",
        ],
        "business": [
            "The Q2 strategy deck is in the shared drive under Strategy/2026/Q2 folder.",
            "The client contract is under review by legal and stored in contracts/pending.",
            "The onboarding checklist document is pinned in the team Notion workspace.",
        ],
        "marketing": [
            "The brand guidelines are in the shared drive under Marketing/Brand/2026.",
            "Campaign performance dashboards are accessible in the analytics platform under Campaigns.",
        ],
        "general": [
            "The main configuration is stored in config.yaml at the project root directory.",
            "The project tracking document is shared with all stakeholders via the team drive.",
        ],
    },

    # ── failed_approach ───────────────────────────────────────────────────────
    # Uses "tried/attempted X but abandoned/failed/switched" pattern — the
    # failure marker (abandoned/dropped/failed/switched) distinguishes from
    # 'decision' type which uses "decided to use X" without failure markers.
    "failed_approach": {
        "software": [
            "We tried MongoDB for document storage but abandoned it due to schema validation problems.",
            "We attempted using Celery for background jobs but dropped it after reliability failures.",
            "We tried Elasticsearch for full-text search but abandoned it because of operational overhead.",
            "We tried Docker Swarm for orchestration but migrated away after reliability issues at scale.",
            "We tried using Memcached for caching but switched because it lacked persistence guarantees.",
            "We tried using RabbitMQ for event processing but abandoned it after message routing errors.",
        ],
        "design": [
            "We tried a full-page onboarding overlay but users abandoned it before completing.",
            "We tested icon-only navigation but switched after usability testing revealed poor discovery.",
            "We tried infinite scroll but abandoned it after users reported losing previously seen content.",
        ],
        "research": [
            "We attempted a longitudinal study but cancelled it after dropout rates invalidated the cohort.",
            "We tried phone interviews but switched to in-person sessions after audio quality was unusable.",
            "We tried a 10-point rating scale but abandoned it when anchoring bias skewed every response.",
        ],
        "business": [
            "We tried a channel-sales model but discontinued it after partners failed to generate leads.",
            "We tried annual upfront contracts but abandoned them after the commitment barrier stalled deals.",
            "We tried building the feature in-house but cancelled it after engineering costs exceeded ROI.",
        ],
        "marketing": [
            "We tried influencer marketing but paused it after campaigns returned a CAC five times target.",
            "We tried weekly email cadence but reduced it after unsubscribe rates rose 20 percent.",
        ],
        "general": [
            "We tried the manual approach but abandoned it after it proved too error-prone to scale.",
            "We tried the original method but replaced it after overhead consistently outweighed benefits.",
        ],
    },

    # ── conversation_note ─────────────────────────────────────────────────────
    "conversation_note": {
        "software": [
            "The developer mentioned they prefer local-first tools over cloud-dependent services.",
            "During the session it was noted that the project is still in early beta stage.",
            "The developer clarified they want offline operation as a hard requirement.",
        ],
        "design": [
            "The designer noted that accessibility is a non-negotiable requirement for this project.",
            "During the review, it was noted that the client prefers a minimal visual style.",
            "The team agreed to defer dark mode implementation to the next design sprint.",
        ],
        "research": [
            "The researcher noted that participant recruitment for this cohort took longer than expected.",
            "During the debrief, the team agreed the interview guide needs simplification.",
        ],
        "business": [
            "The stakeholder mentioned that budget approval for Q4 is likely to be delayed.",
            "During the meeting, the client confirmed the timeline is flexible by two weeks.",
        ],
        "marketing": [
            "The CMO noted that brand consistency is the top priority for this campaign.",
            "It was discussed that the agency needs a full brief two weeks before launch.",
        ],
        "general": [
            "The team agreed that Phase 1 scope should stay minimal and focused on core outcomes.",
            "It was noted that the project should prioritize quality over speed at this stage.",
        ],
    },
}

# ---------------------------------------------------------------------------
# Backward-compat flat TYPE_EXEMPLARS — all domain examples concatenated.
# Imported by semantic_classifier.py for the global centroid computation path.
# ---------------------------------------------------------------------------

TYPE_EXEMPLARS: dict[str, list[str]] = {
    t: [ex for exs in domain_map.values() for ex in exs]
    for t, domain_map in EXEMPLARS.items()
}
