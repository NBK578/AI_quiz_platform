from flask import Blueprint, render_template, request
import openai

debate_bp = Blueprint("debate", __name__)

# GPT 응답 생성 함수
def generate_ai_response(prompt, max_tokens=500):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# /debate 경로 - mode 값에 따라 GPT vs GPT 또는 사용자 vs GPT
@debate_bp.route("/debate", methods=["GET", "POST"])
def debate():
    mode = request.args.get("mode")
    
    # GPT vs GPT 토론
    if mode == "gpt_vs_gpt":
        if request.method == "POST":
            topic = request.form["topic"]
            round_count = int(request.form["round_count"])
            result = {"topic": topic, "rounds": [], "strategies": []}

            for i in range(round_count):
                pro_prompt = f"주제: '{topic}'에 대해 찬성 측 입장에서 토론하세요. 명확한 근거를 들어 주장하세요."
                con_prompt = f"주제: '{topic}'에 대해 반대 측 입장에서 토론하세요. 명확한 근거를 들어 주장하세요."

                pro = generate_ai_response(pro_prompt)
                con = generate_ai_response(con_prompt)

                result["rounds"].append({"pro": pro, "con": con})

                if i < round_count - 1:
                    strategy_pro_prompt = f"""이전 찬성 측 주장: {pro}
이전 반대 측 주장: {con}
다음 라운드에서 찬성 측이 반박을 강화하기 위한 전략을 제안하세요."""
                    strategy_con_prompt = f"""이전 찬성 측 주장: {pro}
이전 반대 측 주장: {con}
다음 라운드에서 반대 측이 반박을 강화하기 위한 전략을 제안하세요."""

                    strategy_pro = generate_ai_response(strategy_pro_prompt, max_tokens=300)
                    strategy_con = generate_ai_response(strategy_con_prompt, max_tokens=300)

                    result["strategies"].append({"pro": strategy_pro, "con": strategy_con})

            summary_prompt = f"주제: '{topic}'에 대한 위의 찬반 토론을 요약하세요. 핵심 쟁점을 중심으로 중립적으로 서술하세요."
            result["summary"] = generate_ai_response(summary_prompt, max_tokens=300)

            return render_template("debate.html", mode=mode, result=result)

        return render_template("debate.html", mode=mode)

    # 사용자 vs GPT 토론
    elif mode == "user_vs_gpt":
        if request.method == "POST":
            topic = request.form["topic"]
            user_statement = request.form["user_statement"]

            gpt_prompt = f"""사용자와 '{topic}' 주제로 토론 중입니다.
사용자 주장: {user_statement}
이에 대한 반대 입장에서 토론 응답을 생성하세요. 예의 바른 어조로 조리 있게 작성하세요."""
            gpt_response = generate_ai_response(gpt_prompt)

            return render_template("debate.html", mode=mode, topic=topic, user_statement=user_statement, gpt_response=gpt_response)

        return render_template("debate.html", mode=mode)

    # 잘못된 접근 시 모드 선택 페이지로 리디렉션
    return render_template("debate_mode_select.html")


# /debate_mode 경로 - 모드 선택 페이지
@debate_bp.route("/debate_mode", methods=["GET"])
def debate_mode_select():
    return render_template("debate_mode_select.html")
