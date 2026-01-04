# models/Text.py
# Optimized prompts for RAER (5 classes) to improve minority-class (Confusion, Enjoyment)
# while keeping majority classes stable (Neutrality, Distraction).
#
# Design principles applied:
# - Keep Neutral concise & non-attractive (avoid "stealing" other classes)
# - Make Enjoyment more "active positive engagement" (clear contrast vs Neutral)
# - Give Confusion strong, unique "cognitive struggle" anchors (NOT distraction)
# - Keep Fatigue distinct (yawn/droop)
# - Make Distraction clearly "external disengagement" (phone/away gaze), not confused focus

# =========================
# Class names (display)
# =========================
class_names_5 = [
    "Neutrality in learning state.",
    "Enjoyment in learning state.",
    "Confusion in learning state.",
    "Fatigue in learning state.",
    "Distraction.",
]

# Slight context wrapper (optional)
class_names_with_context_5 = [
    "an expression of Neutrality in learning state.",
    "an expression of Enjoyment in learning state.",
    "an expression of Confusion in learning state.",
    "an expression of Fatigue in learning state.",
    "an expression of Distraction.",
]

# =========================
# Descriptors (optimized)
# =========================

# Face-only (if you ever switch to only-face setting)
class_descriptor_5_only_face = [
    # Neutral: stable, minimal signal (avoid being too "attractive")
    "Relaxed mouth, open eyes, neutral eyebrows, smooth forehead, steady gaze.",

    # Enjoyment: visible positive affect + active engagement cues
    "Clear smile with raised cheeks, bright eyes, relaxed eyebrows, attentive gaze, occasional nodding.",

    # Confusion: unique anchors of cognitive struggle (focus + tension, gaze switching)
    "Furrowed eyebrows, tightened lips or slightly open mouth, squinting eyes, puzzled gaze shifting between points, tense forehead.",

    # Fatigue: distinct low-energy cues
    "Yawning mouth, drooping eyelids, heavy eyes, slow blinking, head tilts forward.",

    # Distraction: external disengagement cues (away gaze), not tense focus
    "Averted gaze away from task, wandering eyes, neutral mouth, frequent looking aside or down, unfocused face.",
]

# With-context descriptors (recommended for your current pipeline)
class_descriptor_5 = [
    # Neutral: simple, task-present, not overly "good"
    "Relaxed mouth, open eyes, neutral eyebrows, calm expression, looking at study materials with steady posture.",

    # Enjoyment: ACTIVE positive learning engagement (contrast vs Neutral)
    "Smiling mouth corners, bright eyes, relaxed eyebrows, leaning in toward learning content, nodding or reacting positively while following the lesson.",

    # Confusion: cognitive struggle anchors + still task-oriented (NOT distracted)
    "Furrowed eyebrows, tense forehead, puzzled gaze switching between screen and notes, brief hesitation, hand supporting chin or touching forehead while trying to understand.",

    # Fatigue: clear fatigue anchors, not confusion
    "Yawning, drooping eyelids, slow blinking, head tilts forward, reduced movement while still facing the learning material.",

    # Distraction: external disengagement anchors (phone/side gaze), not cognitive struggle
    "Averted gaze from study materials, looking around or down at phone, restless shifting posture, unfocused expression, attention pulled away from the lesson.",
]

# =========================
# Hierarchical Prompts (Lite-HiCroPL)
# =========================
def get_hierarchical_prompts():
    """
    Returns a dictionary of prompts with 3 semantic levels for ensemble.
    Levels:
    1. Visual Primitives: Detailed physical facial features (Anatomical/Forensic).
    2. Behavioral Actions: Observable actions/body language.
    3. Abstract Emotion: High-level emotional state description.
    """
    
    # Level 1: Visual Primitives (Anatomical/Forensic Features)
    level1_visual = [
        "Smooth forehead without wrinkles, eyelids naturally open with iris centered, lips touching lightly without tension.", # Neutral (No muscle activation)
        "Activation of orbicularis oculi (crow's feet) and zygomaticus major (raised cheeks), mouth corners pulled up.", # Enjoyment (Duchenne smile)
        "Vertical glabella lines (frown lines) between eyebrows, inner eyebrows pulled together and down, narrowed eye aperture.", # Confusion (Corrugator activation)
        "Ptosis (drooping upper eyelids) covering part of pupil, slack jaw, heavy eyes with slow blink rate.", # Fatigue (Loss of muscle tone)
        "Saccadic eye movements shifting rapidly, pupils directed laterally away from screen, head rotated away from axis." # Distraction (Visual disengagement)
    ]

    # Level 2: Behavioral Actions (Specific Gestures/Postures)
    level2_action = [
        "Maintaining a static upright posture, eyes tracking the screen content steadily, taking notes rhythmically.", # Neutral
        "Nodding head repeatedly in agreement, leaning torso forward towards the screen, clapping or hand gestures of approval.", # Enjoyment
        "Scratching head or temple, biting lower lip, tilting head to the side while staring fixedly, freezing hand movement.", # Confusion
        "Head dropping forward (nodding off) and snapping back, rubbing eyes vigorously, covering a wide open yawn with hand.", # Fatigue
        "Engaging with a smartphone held in hand, looking around the room, talking to others, physically turning body away from desk." # Distraction
    ]

    # Level 3: Abstract Emotion (Contextual State)
    level3_abstract = [
        "A student maintaining normal attention and baseline composure.",       # Neutral
        "A student exhibiting high engagement, interest, and positive affect.", # Enjoyment
        "A student experiencing cognitive dissonance, difficulty, and puzzlement.", # Confusion
        "A student suffering from exhaustion, drowsiness, and low energy.",     # Fatigue
        "A student completely disengaged from the learning task, focusing on external stimuli." # Distraction
    ]

    return {
        "level1": level1_visual,
        "level2": level2_action,
        "level3": level3_abstract
    }


# ======================================================================
# Helper: return (class_names, input_text) for the model
# ======================================================================
def get_class_info(args):
    """
    Returns:
        class_names: short class names for display/metrics
        input_text : prompt texts for PromptLearner/TextEncoder

    args.text_type options:
        - 'class_names'
        - 'class_names_with_context'
        - 'class_descriptor'
    """

    dataset = getattr(args, "dataset", "RAER")
    if dataset.upper() != "RAER":
        raise ValueError(f"get_class_info currently supports RAER only, got '{dataset}'")

    text_type = getattr(args, "text_type", "class_descriptor")

    if text_type == "class_names":
        class_names = class_names_5
        input_text = class_names_5

    elif text_type == "class_names_with_context":
        class_names = class_names_5
        input_text = class_names_with_context_5

    elif text_type == "class_descriptor":
        class_names = class_names_5
        input_text = class_descriptor_5

    else:
        raise ValueError(f"Unknown text_type: {text_type}")

    return class_names, input_text