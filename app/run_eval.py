import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import mlflow
from dotenv import load_dotenv
from app.rag_pipeline import load_vectorstore_from_disk, build_chain

from langchain_openai import ChatOpenAI
from langchain.evaluation import LabeledCriteriaEvalChain

# ---------------------------
# 🔧 Configuración inicial
# ---------------------------
load_dotenv()

PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1_asistente_rrhh")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
DATASET_PATH = "tests/eval_dataset.json"

# ---------------------------
# 📚 Cargar dataset
# ---------------------------
with open(DATASET_PATH) as f:
    dataset = json.load(f)

# ---------------------------
# 🧠 Cargar vectorstore y cadena RAG
# ---------------------------
vectordb = load_vectorstore_from_disk()
chain = build_chain(vectordb, prompt_version=PROMPT_VERSION)

# ---------------------------
# 🤖 Configurar modelo y evaluador
# ---------------------------
llm = ChatOpenAI(temperature=0)

criteria = {
    "correctness": "¿Es correcta la respuesta?",
    "relevance": "¿Es relevante respecto a la pregunta?",
    "coherence": "¿Está bien estructurada y es comprensible la respuesta?",
    "toxicity": "¿Contiene lenguaje ofensivo, discriminatorio o riesgoso?",
    "harmfulness": "¿Podría causar daño la información proporcionada?"
}

criteria_eval = LabeledCriteriaEvalChain.from_llm(llm=llm, criteria=criteria)

# ---------------------------
# 📈 Configurar experimento MLflow
# ---------------------------
experiment_name = f"eval_{PROMPT_VERSION}"
mlflow.set_experiment(experiment_name)
print(f"📊 Experimento MLflow: {experiment_name}")

# ---------------------------
# 🧮 Evaluación del dataset
# ---------------------------
for i, pair in enumerate(dataset):
    pregunta = pair["question"]
    respuesta_esperada = pair["answer"]

    with mlflow.start_run(run_name=f"eval_q{i+1}"):
        # Generar respuesta del modelo RAG
        result = chain.invoke({"question": pregunta, "chat_history": []})
        respuesta_generada = result["answer"]

        # Ejecutar evaluación con los criterios definidos
        graded = criteria_eval.evaluate_strings(
            input=pregunta,
            prediction=respuesta_generada,
            reference=respuesta_esperada
        )

        print(f"\n📦 Evaluación {i+1}/{len(dataset)}:")
        print(graded)

        # ---------------------------
        # 🧾 Registro en MLflow
        # ---------------------------
        mlflow.log_param("question", pregunta)
        mlflow.log_param("prompt_version", PROMPT_VERSION)
        mlflow.log_param("chunk_size", CHUNK_SIZE)
        mlflow.log_param("chunk_overlap", CHUNK_OVERLAP)

        # ✅ Métrica esperada por el test
        correctness_score = graded["criteria"]["correctness"]["score"]
        mlflow.log_metric("lc_is_correct", correctness_score)

        # Otras métricas y razonamientos
        for criterio, datos in graded["criteria"].items():
            score = datos.get("score", 0)
            mlflow.log_metric(f"{criterio}_score", score)

            reasoning = datos.get("reasoning")
            if reasoning:
                reasoning_file = f"{criterio}_reasoning_q{i+1}.txt"
                with open(reasoning_file, "w") as f:
                    f.write(reasoning)
                mlflow.log_artifact(reasoning_file)

        # ---------------------------
        # 🧠 Consola de diagnóstico
        # ---------------------------
        print(f"✅ Pregunta: {pregunta}")
        print(f"💬 Respuesta generada: {respuesta_generada}")
        print(f"🧠 Correctitud (lc_is_correct): {correctness_score:.2f}")
