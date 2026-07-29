# User onboarding flow states
STATE_IDLE = "idle"
STATE_AWAITING_FILE = "awaiting_file"
STATE_AWAITING_TOKEN_A = "awaiting_token_a"
STATE_AWAITING_TOKEN_B = "awaiting_token_b"
STATE_READY = "ready"

# Active orchestrators: session_id -> ConversationOrchestrator
active_orchestrators: dict = {}
