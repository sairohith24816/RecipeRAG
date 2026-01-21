from typing import List
import os
import time

from dotenv import load_dotenv
from google.generativeai.generative_models import GenerativeModel
from google.generativeai.client import configure
from google.generativeai.types import GenerationConfig


def _configure_gemini() -> None:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not found. Add it to your .env file.")
    configure(api_key=api_key)


def _model_name(config: dict) -> str:
    return config.get("llm_model") or config.get("gemini_model") or "gemini-pro"


def build_rag_prompt(query: str, contexts: List[str], system_instructions: str | None, max_context_chars: int) -> str:
    sys_text = system_instructions or (
        "You are a helpful cooking assistant. Answer strictly using the provided recipes. "
        "If the context is insufficient, say so and ask for clarification."
    )

    packed, used = [], 0
    for c in contexts:
        if not c:
            continue
        remaining = max_context_chars - used
        if remaining <= 0:
            break
        snippet = c if len(c) <= remaining else c[:remaining]
        packed.append(snippet)
        used += len(snippet)

    # Extract target servings if mentioned in query
    scaling_note = ""
    query_lower = query.lower()
    if "people" in query_lower or "servings" in query_lower or "persons" in query_lower:
        import re
        numbers = re.findall(r'\b(\d+)\s*(?:people|persons|servings)', query_lower)
        if numbers:
            target_servings = numbers[0]
            scaling_note = f"\n\nCRITICAL INSTRUCTION: The user needs this recipe for {target_servings} people. You MUST provide specific quantities for EVERY ingredient (e.g., '2kg paneer', '1kg tomatoes', '4 large onions', '200g butter', '100ml cream', '2 tbsp garam masala', etc.). Even if the recipe doesn't list quantities, use your culinary knowledge to estimate realistic amounts for {target_servings} servings. NEVER write 'increase proportionally' or 'adjust to taste' - always give concrete measurements."
    
    return (
        f"System Instructions:\n{sys_text}\n\n"
        f"User Query:\n{query}\n\n"
        f"Context (Recipes):\n" + "\n\n---\n\n".join(packed)
        + f"\n\nTask: Provide a concise, actionable answer with specific ingredient quantities and cooking steps.{scaling_note}"
    )


def _try_extract_text(resp) -> str:
    # Safe access to resp.text
    try:
        t = resp.text  # type: ignore[attr-defined]
        if isinstance(t, str) and t.strip():
            return t.strip()
    except Exception:
        pass

    # Walk candidates/parts
    try:
        candidates = getattr(resp, "candidates", []) or []
        for cand in candidates:
            content = getattr(cand, "content", None) if hasattr(cand, "content") else cand.get("content") if isinstance(cand, dict) else None
            if not content:
                continue
            parts = getattr(content, "parts", None) if hasattr(content, "parts") else content.get("parts") if isinstance(content, dict) else None
            if not parts:
                continue
            chunks = []
            for p in parts:
                text = getattr(p, "text", None) if not isinstance(p, dict) else p.get("text")
                if text:
                    chunks.append(text)
            if chunks:
                return "\n".join(chunks).strip()
    except Exception:
        pass
    return ""


def rank_recipes(query: str, recipes: List[str], config: dict) -> int:
    """
    Ask the LLM to rank the provided recipes based on relevance to the query.
    Returns the index (0-based) of the best recipe.
    """
    _configure_gemini()
    model_name = _model_name(config)
    model = GenerativeModel(model_name)
    
    # Build a ranking prompt with numbered recipes
    recipe_list = []
    for i, recipe in enumerate(recipes, start=1):
        # Truncate each recipe to avoid huge prompts
        truncated = recipe[:1500] if len(recipe) > 1500 else recipe
        recipe_list.append(f"Recipe {i}:\n{truncated}")
    
    ranking_prompt = (
        f"User Query: {query}\n\n"
        "Below are multiple recipes. Rank them by relevance to the user's query. "
        "Respond with ONLY the number (1, 2, 3, etc.) of the MOST relevant recipe.\n\n"
        + "\n\n---\n\n".join(recipe_list)
        + "\n\nYour answer (just the number):"
    )
    
    for attempt in range(3):
        try:
            resp = model.generate_content(
                ranking_prompt,
                generation_config=GenerationConfig(temperature=0.0, max_output_tokens=10),
            )
            text = _try_extract_text(resp).strip()
            # Extract first digit from response
            for char in text:
                if char.isdigit():
                    rank_num = int(char)
                    if 1 <= rank_num <= len(recipes):
                        print(f"[DEBUG] LLM ranked Recipe {rank_num} as top choice.")
                        return rank_num - 1  # Convert to 0-based index
            print(f"[DEBUG] Could not parse ranking from LLM response: '{text}'. Using first recipe.")
            break
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                wait_time = 2 ** attempt
                print(f"[DEBUG] Rate limit hit. Waiting {wait_time}s before retry {attempt+1}/3...")
                time.sleep(wait_time)
            else:
                print(f"[DEBUG] Ranking failed: {e}. Using first recipe.")
                break
    
    return 0  # Default to first recipe if ranking fails


def generate_answer(query: str, retrieved_texts: List[str], config: dict) -> str:
    _configure_gemini()

    model_name = _model_name(config)
    temperature = float(config.get("llm_temperature", 0.2))
    max_output_tokens = int(config.get("llm_max_output_tokens", 512))
    max_context_chars = int(config.get("rag_max_context_chars", 18000))
    system_instructions = config.get("llm_system_instructions")

    model = GenerativeModel(model_name)

    def attempt(max_chars: int, temp: float, max_tokens: int, max_retries: int = 3) -> str:
        prompt = build_rag_prompt(query, retrieved_texts, system_instructions, max_chars)
        print(f"[DEBUG] Prompt length: {len(prompt)} chars, sending to {model_name}...")
        
        for retry in range(max_retries):
            try:
                resp = model.generate_content(
                    prompt,
                    generation_config=GenerationConfig(
                        temperature=temp,
                        max_output_tokens=max_tokens,
                    ),
                )
                print(f"[DEBUG] Response received. Candidates: {len(getattr(resp, 'candidates', []))}")
            except Exception as e:
                error_str = str(e)
                if ("429" in error_str or "quota" in error_str.lower()) and retry < max_retries - 1:
                    wait_time = 2 ** retry
                    print(f"[DEBUG] Rate limit. Waiting {wait_time}s (retry {retry+1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                print(f"[DEBUG] generate_content raised: {e}")
                return ""
            
            try:
                text = _try_extract_text(resp)
                if not text:
                    # Try to show finish_reason for diagnosis
                    try:
                        cands = getattr(resp, "candidates", [])
                        if cands:
                            fr = getattr(cands[0], "finish_reason", None)
                            print(f"[DEBUG] No text extracted. finish_reason={fr}")
                    except Exception:
                        pass
                return text
            except Exception as e:
                print(f"[DEBUG] _try_extract_text raised: {e}")
                return ""
        
        return ""

    # Attempt 1
    text = attempt(max_context_chars, temperature, max_output_tokens)
    if text:
        return text

    # Attempt 2: smaller prompt, safer decoding
    text = attempt(max(4000, int(max_context_chars * 0.5)), min(temperature, 0.2), max_output_tokens + 128)
    if text:
        return text

    return ""
