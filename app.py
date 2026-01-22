"""
AI Verification System - Web UI
Flask backend for the three-way AI verification system with multi-model support
"""
import os
import json
from flask import Flask, render_template, request, jsonify, Response
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# ================= Configuration =================
# Load API keys from environment variables
# Gemini (Google)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_GEMINI = "gemini-2.5-flash"

# Grok (xAI)
GROK_API_KEY = os.getenv("GROK_API_KEY")
MODEL_GROK = "grok-4-1-fast-reasoning"

# OpenAI (Arbiter - fixed)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_OPENAI = "gpt-4o-mini"

# Doubao 豆包 (ByteDance/Volcengine)
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY")  # Replace with your Volcengine API key
DOUBAO_ENDPOINT_ID = os.getenv("DOUBAO_ENDPOINT_ID")  # Replace with your endpoint ID from Volcengine console
MODEL_DOUBAO = "doubao-pro-32k"  # Model name (endpoint ID is used in base_url)

# Qwen 千问 (Alibaba/DashScope)
QWEN_API_KEY = os.getenv("QWEN_API_KEY")  # Replace with your DashScope API key
MODEL_QWEN = "qwen-max"

MAX_REVISION_ROUNDS = 3
# =================================================


def validate_api_keys():
    """Validate required API keys are configured at startup"""
    required_keys = {
        'GEMINI_API_KEY': GEMINI_API_KEY,
        'GROK_API_KEY': GROK_API_KEY,
        'OPENAI_API_KEY': OPENAI_API_KEY,
    }
    
    optional_keys = {
        'DOUBAO_API_KEY': DOUBAO_API_KEY,
        'QWEN_API_KEY': QWEN_API_KEY,
    }
    
    missing = [k for k, v in required_keys.items() if not v]
    if missing:
        raise ValueError(
            f"Missing required API keys: {', '.join(missing)}. "
            "Copy .env.example to .env and configure your keys."
        )
    
    unconfigured = [k for k, v in optional_keys.items() if not v]
    if unconfigured:
        print(f"⚠️ Warning: Optional API keys not configured: {', '.join(unconfigured)}")


# Validate API keys before initializing clients
validate_api_keys()

# Available models configuration
AVAILABLE_MODELS = {
    'gemini': {
        'name': 'Gemini',
        'display_name': 'Gemini (Google)',
        'icon': 'G',
        'color': '#4285f4'
    },
    'grok': {
        'name': 'Grok',
        'display_name': 'Grok (xAI)',
        'icon': 'X',
        'color': '#ff6b35'
    },
    'doubao': {
        'name': 'Doubao',
        'display_name': '豆包 (ByteDance)',
        'icon': '豆',
        'color': '#00d4aa'
    },
    'qwen': {
        'name': 'Qwen',
        'display_name': '千问 (Alibaba)',
        'icon': '千',
        'color': '#ff5722'
    }
}

# Configure clients
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(MODEL_GEMINI)

grok_client = OpenAI(
    api_key=GROK_API_KEY,
    base_url="https://api.x.ai/v1",
)

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Doubao client (OpenAI-compatible)
doubao_client = OpenAI(
    api_key=DOUBAO_API_KEY,
    base_url=f"https://ark.cn-beijing.volces.com/api/v3",
)

# Qwen client (OpenAI-compatible via DashScope)
qwen_client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def ask_gemini(prompt):
    """Request answer from Gemini. Returns (success: bool, content: str)"""
    try:
        response = gemini_model.generate_content(prompt)
        return True, response.text
    except Exception as e:
        return False, f"Gemini error: {e}"


def ask_grok(user_message):
    """Request answer from Grok. Returns (success: bool, content: str)"""
    try:
        response = grok_client.chat.completions.create(
            model=MODEL_GROK,
            messages=[{"role": "user", "content": user_message}],
        )
        return True, response.choices[0].message.content
    except Exception as e:
        return False, f"Grok error: {e}"


def ask_doubao(user_message):
    """Request answer from Doubao. Returns (success: bool, content: str)"""
    try:
        response = doubao_client.chat.completions.create(
            model=DOUBAO_ENDPOINT_ID,  # Use endpoint ID for Volcengine
            messages=[{"role": "user", "content": user_message}],
        )
        return True, response.choices[0].message.content
    except Exception as e:
        return False, f"Doubao error: {e}"


def ask_qwen(user_message):
    """Request answer from Qwen. Returns (success: bool, content: str)"""
    try:
        response = qwen_client.chat.completions.create(
            model=MODEL_QWEN,
            messages=[{"role": "user", "content": user_message}],
        )
        return True, response.choices[0].message.content
    except Exception as e:
        return False, f"Qwen error: {e}"


def ask_openai(prompt):
    """Request answer from OpenAI (Arbiter). Returns (success: bool, content: str)"""
    try:
        response = openai_client.chat.completions.create(
            model=MODEL_OPENAI,
            messages=[
                {"role": "system", "content": "你是一个严格的事实核查仲裁者。你的判断必须准确，尤其要识别AI编造虚假信息（幻觉）的情况。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )
        return True, response.choices[0].message.content
    except Exception as e:
        return False, f"OpenAI error: {e}"


def ask_model(model_id, prompt):
    """Generic function to ask any supported model. Returns (success: bool, content: str)"""
    if model_id == 'gemini':
        return ask_gemini(prompt)
    elif model_id == 'grok':
        return ask_grok(prompt)
    elif model_id == 'doubao':
        return ask_doubao(prompt)
    elif model_id == 'qwen':
        return ask_qwen(prompt)
    else:
        return False, f"Unknown model: {model_id}"


def get_model_name(model_id):
    """Get display name for a model"""
    return AVAILABLE_MODELS.get(model_id, {}).get('name', model_id)


def create_verification_prompt(question, answer):
    """Build verification prompt"""
    return f"""
    我有一个问题: "{question}"
    
    另一个 AI 模型给出了以下回答:
    ---
    {answer}
    ---
    
    请作为一名严厉、客观的事实核查员，评价上述回答。
    你需要：
    1. 检查是否有明显的事实性错误（例如错误的日期、地点、人名等历史事实）。
    2. 指出逻辑漏洞。
    3. 如果回答非常完美，请明确说"完美"或"无需修改"。
    
    重要说明：
    - 不同AI有不同的知识截止日期。如果你无法核实某些近期事件，这并不意味着对方编造了信息。
    - 如果对方引用了你不知道的新闻或事件，请不要仅因为你不知道就判定为"编造"。
    - 只有当你能确认信息是错误的（而不是仅仅是你不知道的），才能指出事实性错误。
    - 如果你没有足够信息来验证，请说明"无法验证"而不是"编造"。
    
    请直接给出你的评审意见。
    """


def create_arbiter_prompt(question, answer, review):
    """Build arbiter prompt"""
    return f"""你是一个公正严格的仲裁者。分析以下评审意见，判断回答是否需要修正。

原始问题: "{question}"

AI 的回答:
---
{answer}
---

另一个 AI 的评审意见:
---
{review}
---

请仔细分析评审意见，判断是否需要修正。

重要判断原则：
1. 不同AI有不同的知识截止日期。如果评审者说"无法找到"或"无法核实"某个事件，这不一定意味着回答是编造的。
2. 只有当评审者能够明确指出具体的事实错误（例如错误的日期、人名、地点等），才应判定需要修订。
3. "无法验证" ≠ "编造"。近期新闻可能在一个AI的知识范围内，但不在另一个AI的知识范围内。
4. 如果评审者仅仅因为自己不知道某信息就指责对方"编造"，这不是有效的批评。
5. 真正需要修订的情况：明确的事实错误、逻辑漏洞、不完整的回答。

请严格按照以下JSON格式回复（不要添加任何其他文字）:

{{"needs_revision": true或false, "reason": "简短说明理由", "corrections": "如需修订，列出具体修正要点；否则为空"}}

注意：如果评审意见只是说"无法核实"或"不知道这个事件"，而没有指出具体错误，则 needs_revision 应为 false。
"""


def create_revision_prompt(question, original_answer, corrections):
    """Build revision prompt"""
    return f"""
针对问题: "{question}"

你之前的回答存在问题:
---
{original_answer}
---

评审发现以下问题需要修正:
---
{corrections}
---

重要修订原则：
1. 如果你编造了不存在的事件或信息，你必须承认错误并明确表示无法核实该事件。
2. 不要再编造任何具体细节（日期、地点、人名等）。
3. 如果无法找到可靠信息，请直接说明而不是编造。

请给出修正后的回答。
"""


def parse_arbiter_response(response):
    """Parse arbiter's response"""
    import re
    
    try:
        result = json.loads(response)
        needs_revision = result.get("needs_revision", False)
        corrections = result.get("corrections", "") or result.get("reason", "")
        return needs_revision, corrections
    except json.JSONDecodeError:
        pass
    
    json_match = re.search(r'\{[^{}]*"needs_revision"[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            needs_revision = result.get("needs_revision", False)
            corrections = result.get("corrections", "") or result.get("reason", "")
            return needs_revision, corrections
        except json.JSONDecodeError:
            pass
    
    response_lower = response.lower()
    if "needs_revision" in response_lower:
        if ": true" in response_lower or ":true" in response_lower:
            return True, response
        elif ": false" in response_lower or ":false" in response_lower:
            return False, ""
    
    if "完美" in response or "无需修改" in response or "no revision" in response_lower:
        return False, ""
    
    return False, ""


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/models')
def get_models():
    """Return available models for the UI"""
    return jsonify(AVAILABLE_MODELS)


@app.route('/verify', methods=['POST'])
def verify():
    """Main verification endpoint - returns streaming updates"""
    question = request.json.get('question', '')
    model_a = request.json.get('model_a', 'gemini')
    model_b = request.json.get('model_b', 'grok')
    
    if not question.strip():
        return jsonify({'error': 'Please enter a question'}), 400
    
    # Validate models
    if model_a not in AVAILABLE_MODELS:
        model_a = 'gemini'
    if model_b not in AVAILABLE_MODELS:
        model_b = 'grok'
    
    model_a_name = get_model_name(model_a)
    model_b_name = get_model_name(model_b)
    
    def generate():
        # Send model info first
        yield json.dumps({
            'type': 'models',
            'model_a': model_a,
            'model_b': model_b,
            'model_a_info': AVAILABLE_MODELS[model_a],
            'model_b_info': AVAILABLE_MODELS[model_b]
        }) + '\n'
        
        # Step 1: Get initial answers
        yield json.dumps({'type': 'status', 'message': f'Getting {model_a_name} response...'}) + '\n'
        success_a, answer_a = ask_model(model_a, question)
        if not success_a:
            yield json.dumps({'type': 'error', 'source': 'model_a', 'message': answer_a}) + '\n'
            return
        yield json.dumps({'type': 'model_a_initial', 'content': answer_a}) + '\n'
        
        yield json.dumps({'type': 'status', 'message': f'Getting {model_b_name} response...'}) + '\n'
        success_b, answer_b = ask_model(model_b, question)
        if not success_b:
            yield json.dumps({'type': 'error', 'source': 'model_b', 'message': answer_b}) + '\n'
            return
        yield json.dumps({'type': 'model_b_initial', 'content': answer_b}) + '\n'
        
        # Verification loop
        round_num = 1
        consensus = False
        
        while round_num <= MAX_REVISION_ROUNDS and not consensus:
            yield json.dumps({'type': 'status', 'message': f'Round {round_num}: Cross-verification in progress...'}) + '\n'
            
            # Model B reviews Model A
            yield json.dumps({'type': 'status', 'message': f'Round {round_num}: {model_b_name} reviewing {model_a_name}...'}) + '\n'
            success_review_b, review_b = ask_model(model_b, create_verification_prompt(question, answer_a))
            if not success_review_b:
                yield json.dumps({'type': 'error', 'source': 'model_b', 'message': review_b}) + '\n'
                return
            yield json.dumps({'type': 'review', 'reviewer': 'model_b', 'target': 'model_a', 'round': round_num, 'content': review_b}) + '\n'
            
            # Model A reviews Model B
            yield json.dumps({'type': 'status', 'message': f'Round {round_num}: {model_a_name} reviewing {model_b_name}...'}) + '\n'
            success_review_a, review_a = ask_model(model_a, create_verification_prompt(question, answer_b))
            if not success_review_a:
                yield json.dumps({'type': 'error', 'source': 'model_a', 'message': review_a}) + '\n'
                return
            yield json.dumps({'type': 'review', 'reviewer': 'model_a', 'target': 'model_b', 'round': round_num, 'content': review_a}) + '\n'
            
            # OpenAI arbiter analysis
            yield json.dumps({'type': 'status', 'message': f'Round {round_num}: OpenAI arbiter analyzing...'}) + '\n'
            
            success_arb_a, arbiter_a = ask_openai(create_arbiter_prompt(question, answer_a, review_b))
            if not success_arb_a:
                yield json.dumps({'type': 'error', 'source': 'arbiter', 'message': arbiter_a}) + '\n'
                return
            a_needs_revision, a_corrections = parse_arbiter_response(arbiter_a)
            
            success_arb_b, arbiter_b = ask_openai(create_arbiter_prompt(question, answer_b, review_a))
            if not success_arb_b:
                yield json.dumps({'type': 'error', 'source': 'arbiter', 'message': arbiter_b}) + '\n'
                return
            b_needs_revision, b_corrections = parse_arbiter_response(arbiter_b)
            
            arbiter_summary = {
                'round': round_num,
                'model_a_analysis': arbiter_a,
                'model_b_analysis': arbiter_b,
                'model_a_needs_revision': a_needs_revision,
                'model_b_needs_revision': b_needs_revision
            }
            yield json.dumps({'type': 'arbiter', **arbiter_summary}) + '\n'
            
            if not a_needs_revision and not b_needs_revision:
                consensus = True
                yield json.dumps({'type': 'consensus', 'reached': True, 'round': round_num}) + '\n'
            else:
                # Revisions needed
                if a_needs_revision:
                    yield json.dumps({'type': 'status', 'message': f'Round {round_num}: {model_a_name} revising answer...'}) + '\n'
                    success_rev_a, answer_a = ask_model(model_a, create_revision_prompt(question, answer_a, a_corrections))
                    if not success_rev_a:
                        yield json.dumps({'type': 'error', 'source': 'model_a', 'message': answer_a}) + '\n'
                        return
                    yield json.dumps({'type': 'model_a_revised', 'round': round_num, 'content': answer_a}) + '\n'
                
                if b_needs_revision:
                    yield json.dumps({'type': 'status', 'message': f'Round {round_num}: {model_b_name} revising answer...'}) + '\n'
                    success_rev_b, answer_b = ask_model(model_b, create_revision_prompt(question, answer_b, b_corrections))
                    if not success_rev_b:
                        yield json.dumps({'type': 'error', 'source': 'model_b', 'message': answer_b}) + '\n'
                        return
                    yield json.dumps({'type': 'model_b_revised', 'round': round_num, 'content': answer_b}) + '\n'
                
                round_num += 1
        
        if not consensus:
            yield json.dumps({'type': 'consensus', 'reached': False, 'round': round_num - 1}) + '\n'
        
        # Final results
        yield json.dumps({'type': 'final', 'model_a': answer_a, 'model_b': answer_b, 'consensus': consensus}) + '\n'
    
    return Response(generate(), mimetype='application/x-ndjson')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
