import streamlit as st
import sqlite3
import os
import base64
from PIL import Image
from io import BytesIO
import requests
from bs4 import BeautifulSoup
import numpy as np
import re
import pandas as pd

# IMPORTAR SOMENTE de dentro de tensorflow.keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D as _BaseDepthwiseConv2D

import google.generativeai as genai
import streamlit.components.v1 as components
import dotenv

# --- Carregamento das variáveis de ambiente (se existir arquivo .env) ---
dotenv.load_dotenv()

# --- Configura a chave da API do Google Gemini ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "Sua_Chave_Google_Aqui")
genai.configure(api_key=GOOGLE_API_KEY)

# ——— Subclasse para “ignorar” o parâmetro groups ao carregar o .h5 ———
class DepthwiseConv2D(_BaseDepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        # Remove o argumento 'groups' caso exista, para não quebrar o construtor original
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

# ——— Carrega o modelo “FV.h5” usando custom_objects para DepthwiseConv2D ———
# Certifique-se de que “FV.h5” esteja na mesma pasta deste script
modelo = load_model(
    'FV.h5',
    custom_objects={'DepthwiseConv2D': DepthwiseConv2D}
)

# Rótulos para as previsões
rotulos = {
    0: 'maçã', 1: 'banana', 2: 'beterraba', 3: 'pimentão', 4: 'repolho', 5: 'capsicum', 6: 'cenoura',
    7: 'couve-flor', 8: 'pimenta', 9: 'milho', 10: 'pepino', 11: 'berinjela', 12: 'alho', 13: 'gengibre',
    14: 'uvas', 15: 'jalapeno', 16: 'kiwi', 17: 'limão', 18: 'alface',
    19: 'manga', 20: 'cebola', 21: 'laranja', 22: 'paprika', 23: 'pera', 24: 'ervilhas', 25: 'abacaxi',
    26: 'romã', 27: 'batata', 28: 'rabanete', 29: 'soja', 30: 'espinafre', 31: 'milho-doce',
    32: 'batata-doce', 33: 'tomate', 34: 'nabo', 35: 'melancia'
}

# Definir categorias de frutas e vegetais
frutas = [
    'maçã', 'banana', 'pimentão', 'pimenta', 'uvas', 'jalapeno', 'kiwi',
    'limão', 'manga', 'laranja', 'paprika', 'pera', 'abacaxi', 'romã', 'melancia'
]
vegetais = [
    'beterraba', 'repolho', 'capsicum', 'cenoura', 'couve-flor', 'milho',
    'pepino', 'berinjela', 'gengibre', 'alface', 'cebola', 'ervilhas',
    'batata', 'rabanete', 'soja', 'espinafre', 'milho-doce', 'batata-doce',
    'tomate', 'nabo'
]

# ——— Carrega o Excel de nutrientes para fallback caso o scraping falhe ———
EXCEL_PATH = "frutas_vegetais_nutrientes_completo.xlsx"
try:
    df_nutrientes = pd.read_excel(EXCEL_PATH)
    # Espera-se que o arquivo tenha colunas como:
    # "Item" (nome da fruta/vegetal), "Calorias (100g)", "Proteínas (g)",
    # "Carboidratos (g)", "Gorduras (g)", "Vitamina C (mg)", "Potássio (mg)"
    # e quaisquer outras colunas de micronutrientes/macronutrientes que você tenha incluído.
except FileNotFoundError:
    st.error(f"Arquivo '{EXCEL_PATH}' não encontrado. Coloque-o na mesma pasta do app.")
    df_nutrientes = pd.DataFrame(columns=[
        'Item', 'Calorias (100g)', 'Proteínas (g)', 'Carboidratos (g)',
        'Gorduras (g)', 'Vitamina C (mg)', 'Potássio (mg)'
    ])

# ——— Função para buscar informações de calorias: Google → Regex → Excel ———
def buscar_calorias(predicao: str) -> str | None:
    """
    1) Tenta obter o valor calórico buscando no Google via scraping.
    2) Se encontrar, retorna algo como "89 kcal".
    3) Se não houver resultado confiável, faz fallback no DataFrame lido do Excel.
       Neste caso, retorna apenas o valor de calorias do Excel (mas não exibe outros nutrientes aqui).
    """
    nome = predicao.strip()
    try:
        # Monta a URL de busca no Google
        query = nome.replace(" ", "+")
        url = f"https://www.google.com/search?q=calorias+em+{query}"
        resp = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }, timeout=5).text
        soup = BeautifulSoup(resp, "html.parser")

        # 1) Tentar seletor antigo que às vezes funciona
        calorias_div = soup.find("div", class_="BNeawe iBp4i AP7Wnd")
        if calorias_div and ("kcal" in calorias_div.text.lower() or "caloria" in calorias_div.text.lower()):
            return calorias_div.text.strip()

        # 2) Tentar outro seletor em cards: classe "Z0LcW"
        calorias_card = soup.find("div", class_="Z0LcW")
        if calorias_card and ("kcal" in calorias_card.text.lower() or "caloria" in calorias_card.text.lower()):
            return calorias_card.text.strip()

        # 3) Se ainda nada, varrer TODO o texto em busca de padrão "123 kcal" ou "123 calorias"
        texto_completo = soup.get_text(separator=" ", strip=True)
        padrao = re.search(
            r"\b(\d+(?:[.,]\d+)*)\s*(kcal|caloria(?:s)?)\b",
            texto_completo, re.IGNORECASE
        )
        if padrao:
            return padrao.group(0).capitalize()

    except Exception:
        # Qualquer erro no scraping/requests, segue para o fallback no Excel
        pass

    # 4) Fallback Excel apenas para calorias
    try:
        row = df_nutrientes.loc[df_nutrientes["Item"].str.lower() == nome.lower()]
        if not row.empty:
            calorias_val = row.iloc[0]["Calorias (100g)"]
            if not (pd.isna(calorias_val)):
                if float(calorias_val).is_integer():
                    return f"{int(calorias_val)} kcal"
                else:
                    return f"{calorias_val} kcal"
    except Exception:
        pass

    st.warning("Informação de calorias não disponível.")
    return None

# ——— Função para obter todos os nutrientes a partir do Excel ———
def obter_todos_nutrientes(predicao: str) -> dict | None:
    """
    Retorna um dicionário com todos os nutrientes para a fruta/vegetal
    encontrada no Excel. Se não achar, retorna None.
    """
    nome = predicao.strip()
    try:
        row = df_nutrientes.loc[df_nutrientes["Item"].str.lower() == nome.lower()]
        if not row.empty:
            # Converte a linha em um dicionário (Item → valor)
            dados = row.iloc[0].to_dict()
            return dados
    except Exception:
        pass
    return None

# ——— Função para preparar a imagem para previsão (Keras) ———
def preparar_imagem(caminho_img: str) -> str:
    img = load_img(caminho_img, target_size=(224, 224, 3))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)  # formato (1, 224, 224, 3)
    preds = modelo.predict(arr)
    y_class = int(np.argmax(preds, axis=1)[0])
    res = rotulos[y_class]
    return res.capitalize()

# ——— Função para gerar receita via Gemini-Pro ———
def gerar_receita(ingredientes: list[str]) -> str:
    prompt = f"Crie uma receita simples que utilize os seguintes ingredientes: {', '.join(ingredientes)}."
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt, stream=True)
        receita = ""
        for chunk in response:
            if chunk.text:
                receita += chunk.text
        return receita if receita else "Nenhuma receita encontrada."
    except Exception as e:
        st.error(f"Erro ao gerar receita: {e}")
        return "Desculpe, não foi possível gerar uma receita no momento."

# ——— Módulo Offline de Classificação de Frutas/Vegetais ———
def run_offline_classifier():
    st.subheader("Classificação Offline (Frutas/Vegetais)")
    img_file = st.file_uploader(
        "Carregar imagem de fruta ou vegetal (um por vez)",
        type=["jpg", "png", "jpeg"]
    )
    if not img_file:
        return

    # Mostra a imagem redimensionada
    img = Image.open(img_file).resize((250, 250))
    st.image(img, caption="Imagem selecionada", use_container_width=False)

    # Salva localmente
    save_dir = "upload_images"
    os.makedirs(save_dir, exist_ok=True)
    caminho = os.path.join(save_dir, img_file.name)
    with open(caminho, "wb") as f:
        f.write(img_file.getbuffer())

    # Previsão
    resultado = preparar_imagem(caminho)
    categoria = "Vegetal" if resultado.lower() in vegetais else "Fruta"
    st.info(f"**Categoria:** {categoria}")
    st.success(f"**Previsão:** {resultado}")

    # Tenta mostrar calorias via Google → fallback Excel
    calorias = buscar_calorias(resultado)
    if calorias:
        st.warning(f"**{calorias} (por 100g)**")

    # Em seguida, tenta exibir todos os nutrientes do Excel
    todos_nutrientes = obter_todos_nutrientes(resultado)
    if todos_nutrientes:
        # Remove a chave "Item" pois já sabemos o nome
        todos_nutrientes.pop("Item", None)
        st.write("#### Perfil Nutricional completo (por 100g)")
        # Exibe cada nutriente formatado
        for chave, valor in todos_nutrientes.items():
            if pd.notna(valor):
                st.write(f"- **{chave}:** {valor}")
    else:
        st.info("Nenhum dado completo de nutrientes encontrado no arquivo Excel.")

# ——— Funções de manipulação de imagens para Gemini Chat (histórico) ———
def get_image_base64(image_raw: Image.Image) -> str:
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')

def base64_to_image(base64_string: str) -> Image.Image:
    base64_string = base64_string.split(",")[1]
    return Image.open(BytesIO(base64.b64decode(base64_string)))

# ——— Banco de Dados SQLite para usuários e conversas ———
DB_PATH = "app.db"

def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    # Tabela de usuários
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            api_key TEXT,
            idade INTEGER,
            peso REAL,
            altura REAL,
            nivel_atividade TEXT,
            restricoes_alimentares TEXT
        )
    """)
    # Tabela de sessões de chat
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT DEFAULT 'Novo Chat',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    # Tabela de conversas
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(chat_id) REFERENCES chat_sessions(id),
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    update_schema(conn)
    return conn

def update_schema(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(conversations)")
    columns = [col[1] for col in cursor.fetchall()]
    if "chat_id" not in columns:
        cursor.execute("ALTER TABLE conversations ADD COLUMN chat_id INTEGER")
        conn.commit()

conn = init_db()

def register_user(
    username: str, password: str, api_key: str,
    idade: int, peso: float, altura: float,
    nivel_atividade: str, restricoes_alimentares: list[str]
) -> tuple[bool, str]:
    cursor = conn.cursor()
    restricoes_str = ",".join(restricoes_alimentares) if restricoes_alimentares else ""
    try:
        cursor.execute(
            "INSERT INTO users (username, password, api_key, idade, peso, altura, nivel_atividade, restricoes_alimentares) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (username, password, api_key, idade, peso, altura, nivel_atividade, restricoes_str)
        )
        conn.commit()
        return True, "Usuário registrado com sucesso!"
    except sqlite3.IntegrityError:
        return False, "Usuário já existe."

def update_user_health(
    user_id: int, idade: int, peso: float,
    altura: float, nivel_atividade: str, restricoes_alimentares: list[str]
):
    cursor = conn.cursor()
    restricoes_str = ",".join(restricoes_alimentares) if restricoes_alimentares else ""
    cursor.execute(
        "UPDATE users SET idade = ?, peso = ?, altura = ?, nivel_atividade = ?, restricoes_alimentares = ? WHERE id = ?",
        (idade, peso, altura, nivel_atividade, restricoes_str, user_id)
    )
    conn.commit()

def login_user(username: str, password: str) -> dict | None:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, api_key, idade, peso, altura, nivel_atividade, restricoes_alimentares FROM users WHERE username = ? AND password = ?",
        (username, password)
    )
    result = cursor.fetchone()
    if result:
        return {
            "id": result[0],
            "username": username,
            "api_key": result[1],
            "idade": result[2],
            "peso": result[3],
            "altura": result[4],
            "nivel_atividade": result[5],
            "restricoes_alimentares": result[6]
        }
    else:
        return None

def create_chat_session(user_id: int, title: str = "Novo Chat") -> int:
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_sessions (user_id, title) VALUES (?, ?)", (user_id, title))
    conn.commit()
    return cursor.lastrowid

def get_chat_sessions(user_id: int) -> list[tuple]:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, title, timestamp FROM chat_sessions WHERE user_id = ? ORDER BY timestamp DESC",
        (user_id,)
    )
    return cursor.fetchall()

def get_conversation_history(chat_id: int) -> list[dict]:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content, timestamp FROM conversations WHERE chat_id = ? ORDER BY timestamp",
        (chat_id,)
    )
    rows = cursor.fetchall()
    history = []
    for role, content, timestamp in rows:
        history.append({"role": role, "content": content, "timestamp": timestamp})
    return history

def add_message(chat_id: int, user_id: int, role: str, content: str):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO conversations (chat_id, user_id, role, content) VALUES (?, ?, ?, ?)",
        (chat_id, user_id, role, content)
    )
    conn.commit()

# ——— Conversão de mensagens para formato Gemini ———
def messages_to_gemini(messages: list[dict]) -> list[dict]:
    gemini_messages = []
    prev_role = None
    for message in messages:
        if prev_role and (prev_role == message["role"]):
            gemini_message = gemini_messages[-1]
        else:
            gemini_message = {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [],
            }
        for content in message["content"]:
            if content["type"] == "text":
                gemini_message["parts"].append(content["text"])
            elif content["type"] == "image_url":
                gemini_message["parts"].append(base64_to_image(content["image_url"]["url"]))
        if prev_role != message["role"]:
            gemini_messages.append(gemini_message)
        prev_role = message["role"]
    return gemini_messages

# ——— Streaming de resposta de texto simples (Gemini) ———
def stream_llm_response(model_params: dict, api_key: str | None = None, prompt_override: str | None = None):
    response_message = ""
    genai.configure(api_key=api_key)
    model_g = genai.GenerativeModel(
        model_name=model_params["model"],
        generation_config={"temperature": model_params["temperature"]}
    )
    if prompt_override:
        gemini_messages = [{"role": "user", "parts": [prompt_override]}]
    else:
        gemini_messages = messages_to_gemini(st.session_state.messages)

    for chunk in model_g.generate_content(contents=gemini_messages, stream=True):
        chunk_text = chunk.text or ""
        response_message += chunk_text
        yield chunk_text

    st.session_state.messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": response_message}]
    })
    add_message(st.session_state.chat_id, st.session_state.user["id"], "assistant", response_message)

# ——— Streaming multimídia (texto, áudio, vídeo, imagem) ———
def stream_multimedia_realtime_response(api_key: str, prompt_override: str | None = None):
    response_message = ""
    genai.configure(api_key=api_key)
    model_g = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config={"temperature": 0.3}
    )
    if prompt_override:
        gemini_messages = [{"role": "user", "parts": [prompt_override]}]
    else:
        gemini_messages = messages_to_gemini(st.session_state.messages)

    for chunk in model_g.generate_content(contents=gemini_messages, stream=True):
        if chunk.text:
            response_message += chunk.text
            yield chunk.text
        elif hasattr(chunk, 'inline_data') and chunk.inline_data is not None:
            mime = chunk.inline_data.mime_type
            data_b64 = base64.b64encode(chunk.inline_data.data).decode('utf-8')
            if mime == "audio/pcm":
                msg = f"[Áudio recebido: {data_b64[:30]}...]"
            elif mime == "video/mp4":
                msg = f"[Vídeo recebido: {data_b64[:30]}...]"
            elif mime == "image/jpeg":
                msg = f"[Imagem recebida: {data_b64[:30]}...]"
            else:
                msg = f"[Mídia recebida: {data_b64[:30]}...]"
            response_message += msg
            yield msg

    st.session_state.messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": response_message}]
    })
    add_message(st.session_state.chat_id, st.session_state.user["id"], "assistant", response_message)

# ——— Interface HTML do chat realtime via iframe ———
def realtime_chat_interface():
    chat_url = "http://127.0.0.1:3000/"
    st.info("Certifique-se de que o servidor do frontend esteja rodando em http://127.0.0.1:3000/")

    if "show_iframe_full" not in st.session_state:
        st.session_state.show_iframe_full = False

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔳 Abrir em Tela Cheia"):
            st.session_state.show_iframe_full = True

    if st.session_state.show_iframe_full:
        st.markdown("### Modo Tela Cheia")
        st.markdown(f"""
            <iframe src="{chat_url}"
                width="100%"
                height="900"
                frameborder="0"
                allow="microphone; camera; display-capture; autoplay"
                allowfullscreen
                style="border-radius: 10px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);">
            </iframe>
        """, unsafe_allow_html=True)
    else:
        st.markdown("### Visualização Padrão")
        components.html(f"""
            <iframe src="{chat_url}"
                width="100%"
                height="600"
                frameborder="0"
                allow="microphone; camera; display-capture; autoplay"
                allowfullscreen>
            </iframe>
        """, height=600, scrolling=True)

# ——— Funções específicas do aplicativo de nutrição/Chat ———
def analyze_dish_image(
    image: Image.Image,
    google_api_key: str,
    idade: int,
    peso: float,
    altura: float,
    imc: float,
    nivel_atividade: str
):
    # Monta prompt de texto com dados de usuário
    text_prompt = (
        f"Atue como um nutricionista. Aqui estão algumas informações para ajudar a estimar as calorias do prato: "
        f"idade {idade} anos, peso {peso} kg, altura {altura} m, IMC {imc} e nível de atividade física {nivel_atividade}. "
        "Por favor, forneça uma estimativa calórica para este prato com base nas informações fornecidas."
    )
    # Converte imagem em base64 e adiciona à mensagem
    image_b64 = get_image_base64(image)
    st.session_state.messages.append({
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}]
    })
    add_message(st.session_state.chat_id, st.session_state.user["id"], "user", "[Imagem em base64]")
    st.session_state.messages.append({
        "role": "user",
        "content": [{"type": "text", "text": text_prompt}]
    })
    add_message(st.session_state.chat_id, st.session_state.user["id"], "user", text_prompt)

    with st.chat_message("assistant"):
        st.write_stream(
            stream_llm_response(
                {"model": "gemini-2.0-flash", "temperature": 0.3},
                google_api_key
            )
        )

def recommend_recipes_with_ingredients(image: Image.Image, google_api_key: str):
    restricoes = ", ".join(st.session_state.restricoes_alimentares) if st.session_state.restricoes_alimentares else ""
    image_b64 = get_image_base64(image)
    st.session_state.messages.append({
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}]
    })
    add_message(st.session_state.chat_id, st.session_state.user["id"], "user", "[Imagem em base64]")
    st.session_state.messages.append({
        "role": "user",
        "content": [{
            "type": "text",
            "text": f"Baseando-se nos ingredientes da imagem e nas seguintes restrições alimentares: {restricoes}, "
                    "recomende receitas saudáveis para o meu perfil."
        }]
    })
    add_message(
        st.session_state.chat_id,
        st.session_state.user["id"],
        "user",
        f"Baseando-se nos ingredientes da imagem e restrições: {restricoes}"
    )

    with st.chat_message("assistant"):
        st.write_stream(
            stream_llm_response(
                {"model": "gemini-2.0-flash", "temperature": 0.3},
                google_api_key
            )
        )

def generate_shopping_list_recipes(shopping_list: str, days: int, google_api_key: str):
    user_health = (
        f"idade {st.session_state.user.get('idade')}, peso {st.session_state.user.get('peso')} kg, "
        f"altura {st.session_state.user.get('altura')} m, nível de atividade física {st.session_state.user.get('nivel_atividade')}"
    )
    restricoes = st.session_state.user.get('restricoes_alimentares') or "Nenhuma"
    restricoes = restricoes.replace(",", ", ")
    prompt = (
        f"Você é um nutricionista. Tenho a seguinte lista de compras: {shopping_list}. "
        f"Preciso de receitas para os próximos {days} dias. Considere meus dados de saúde: {user_health} "
        f"e minhas restrições alimentares: {restricoes}. "
        "Por favor, elabore uma receita balanceada para cada dia, contando somente com a minha lista de compras."
    )

    st.session_state.messages.append({
        "role": "user",
        "content": [{"type": "text", "text": f"Gerar receitas com base na lista de compras: {shopping_list} para {days} dias."}]
    })
    add_message(
        st.session_state.chat_id,
        st.session_state.user["id"],
        "user",
        f"Gerar receitas com base na lista de compras: {shopping_list} para {days} dias"
    )

    with st.chat_message("assistant"):
        st.write_stream(
            stream_llm_response(
                {"model": "gemini-2.0-flash", "temperature": 0.3},
                google_api_key,
                prompt_override=prompt
            )
        )

# ——— Tela de Login e Cadastro ———
def login_screen():
    st.sidebar.title("Autenticação")
    mode = st.sidebar.radio("Entre ou Cadastre-se", ["Login", "Cadastro"])
    username = st.sidebar.text_input("Usuário")
    password = st.sidebar.text_input("Senha", type="password")
    api_key_input = st.sidebar.text_input("Chave API do Google (opcional)", type="password")

    if mode == "Cadastro":
        st.sidebar.write("### Dados de Saúde do Usuário")
        idade_reg = st.sidebar.number_input("Idade", min_value=1, max_value=120, step=1)
        peso_reg = st.sidebar.number_input("Peso (kg)", min_value=1.0, format="%.2f")
        altura_reg = st.sidebar.number_input("Altura (m)", min_value=0.5, format="%.2f")
        nivel_atividade_reg = st.sidebar.selectbox(
            "Nível de Atividade Física", ["Sedentário", "Moderado", "Ativo", "Muito Ativo"]
        )
        restricoes_alim_reg = st.sidebar.multiselect(
            "Restrições Alimentares",
            ["Diabetes", "Hipertensão", "Alergias Alimentares", "Doenças Celíacas",
             "Vegetariano", "Vegano", "Low Carb", "Keto"]
        )
        if st.sidebar.button("Cadastrar"):
            if username and password:
                success, msg = register_user(
                    username, password, api_key_input,
                    idade_reg, peso_reg, altura_reg,
                    nivel_atividade_reg, restricoes_alim_reg
                )
                st.sidebar.info(msg)
            else:
                st.sidebar.error("Preencha usuário e senha para cadastro.")
    else:
        if st.sidebar.button("Login"):
            user = login_user(username, password)
            if user:
                st.session_state.user = user
                # Armazena chave API fornecida (caso envie uma nova)
                if api_key_input:
                    user["api_key"] = api_key_input
                chat_id = create_chat_session(user["id"])
                st.session_state.chat_id = chat_id
                st.session_state.messages = []
                st.sidebar.success(f"Bem-vindo, {username}!")
            else:
                st.sidebar.error("Usuário ou senha incorretos.")

# ——— Função principal ———
def main():
    st.set_page_config(
        page_title="App Nutrição TCC",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Se não estiver logado, mostra tela de login/cadastro
    if "user" not in st.session_state:
        login_screen()
        st.write("Por favor, realize o login ou cadastro para utilizar o aplicativo.")
        return

    # Inicializa lista de restrições no session_state, se não existir
    if "restricoes_alimentares" not in st.session_state:
        restr = st.session_state.user.get("restricoes_alimentares", "")
        st.session_state.restricoes_alimentares = [r.strip() for r in restr.split(",")] if restr else []

    st.title("App Nutrição TCC 💬")

    # Opções de Chat (menu lateral) e API key
    menu_option = st.sidebar.radio("Menu Chat", ["Chat", "Histórico de Conversas", "Novo Chat"])
    google_api_key = st.text_input(
        "Sua chave API do Google",
        value=st.session_state.user.get("api_key") or GOOGLE_API_KEY,
        type="password"
    )
    st.session_state.user["api_key"] = google_api_key

    # Exibe dados de saúde cadastrados no sidebar
    st.sidebar.divider()
    st.sidebar.write("### Dados de Saúde (Cadastro)")
    st.sidebar.write(f"**Idade:** {st.session_state.user.get('idade', 'N/D')} anos")
    st.sidebar.write(f"**Peso (kg):** {st.session_state.user.get('peso', 'N/D')} kg")
    st.sidebar.write(f"**Altura (m):** {st.session_state.user.get('altura', 'N/D')} m")
    st.sidebar.write(f"**Nível de Atividade:** {st.session_state.user.get('nivel_atividade', 'N/D')}")
    restr_display = st.session_state.restricoes_alimentares
    st.sidebar.write(f"**Restrições Alimentares:** {', '.join(restr_display) if restr_display else 'Nenhuma'}")

    # Expander para alterar dados de saúde
    with st.sidebar.expander("Alterar Dados de Saúde", expanded=False):
        current_idade = st.session_state.user.get("idade") or 25
        current_peso = st.session_state.user.get("peso") or 70.0
        current_altura = st.session_state.user.get("altura") or 1.75
        current_nivel = st.session_state.user.get("nivel_atividade") or "Moderado"
        current_restricoes = st.session_state.restricoes_alimentares

        new_idade = st.number_input(
            "Idade", min_value=1, max_value=120, step=1,
            value=current_idade, key="upd_idade"
        )
        new_peso = st.number_input(
            "Peso (kg)", min_value=1.0, format="%.2f",
            value=current_peso, key="upd_peso"
        )
        new_altura = st.number_input(
            "Altura (m)", min_value=0.5, format="%.2f",
            value=current_altura, key="upd_altura"
        )
        new_nivel = st.selectbox(
            "Nível de Atividade Física",
            ["Sedentário", "Moderado", "Ativo", "Muito Ativo"],
            index=["Sedentário", "Moderado", "Ativo", "Muito Ativo"].index(current_nivel),
            key="upd_nivel"
        )
        new_restricoes = st.multiselect(
            "Restrições Alimentares",
            ["Diabetes", "Hipertensão", "Alergias Alimentares", "Doenças Celíacas",
             "Vegetariano", "Vegano", "Low Carb", "Keto"],
            default=current_restricoes, key="upd_restricoes"
        )
        if st.button("Atualizar Dados de Saúde"):
            update_user_health(
                st.session_state.user["id"],
                new_idade, new_peso, new_altura, new_nivel, new_restricoes
            )
            st.session_state.user["idade"] = new_idade
            st.session_state.user["peso"] = new_peso
            st.session_state.user["altura"] = new_altura
            st.session_state.user["nivel_atividade"] = new_nivel
            st.session_state.user["restricoes_alimentares"] = ",".join(new_restricoes)
            st.session_state.restricoes_alimentares = new_restricoes
            st.success("Dados de saúde atualizados com sucesso!")

    # Área de upload de imagem e seleção de análise
    st.sidebar.divider()
    st.sidebar.write("### Opções de Análise")
    uploaded_image = st.sidebar.file_uploader(
        "Carregar imagem de refeição ou ingredientes:", type=["png", "jpg", "jpeg"]
    )
    option = st.sidebar.selectbox(
        "Escolha a análise desejada",
        [
            "Calcular Calorias do Prato",
            "Recomendar Receitas com Ingredientes",
            "Lista de Compras",
            "Chat Multimídia (Real-time)",
            "Classificação Offline (Frutas/Vegetais)"
        ]
    )

    # Se escolhida Lista de Compras
    if option == "Lista de Compras":
        st.sidebar.subheader("Lista de Compras")
        shopping_list = st.sidebar.text_area(
            "Informe sua lista de compras (itens separados por vírgula ou linha):"
        )
        days = st.sidebar.number_input("Para quantos dias será a lista?", min_value=1, step=1)
        if st.sidebar.button("Gerar Receitas para os Dias"):
            generate_shopping_list_recipes(shopping_list, days, google_api_key)
            return

    # Botão para resetar conversa
    if st.sidebar.button("🗑️ Resetar conversa"):
        st.session_state.messages = []
        cursor = conn.cursor()
        cursor.execute("DELETE FROM conversations WHERE chat_id = ?", (st.session_state.chat_id,))
        conn.commit()

    # Menu de Chat: Histórico / Novo Chat / Conversa
    if menu_option == "Histórico de Conversas":
        st.subheader("Histórico de Conversas")
        sessions = get_chat_sessions(st.session_state.user["id"])
        if sessions:
            for chat in sessions:
                st.write(f"**Chat ID {chat[0]} - {chat[1]}** (Criado em {chat[2]})")
                hist = get_conversation_history(chat[0])
                for msg in hist:
                    st.write(f"**{msg['timestamp']} - {msg['role'].capitalize()}:** {msg['content']}")
                st.write("---")
        else:
            st.write("Nenhuma conversa encontrada.")
        return

    elif menu_option == "Novo Chat":
        new_chat_id = create_chat_session(st.session_state.user["id"])
        st.session_state.chat_id = new_chat_id
        st.session_state.messages = []
        st.success("Novo chat criado com sucesso!")
        st.write("Inicie sua nova conversa...")
        return

    # Se escolher Chat Multimídia
    if option == "Chat Multimídia (Real-time)":
        st.subheader("Chat Multimídia (Real-time)")
        st.info("Certifique-se de que o servidor do frontend esteja rodando em http://127.0.0.1:3000/")
        realtime_chat_interface()
        return

    # Se escolher Classificação Offline
    if option == "Classificação Offline (Frutas/Vegetais)":
        run_offline_classifier()
        return

    # Caso seja Calcular Calorias ou Recomendar Receitas com Ingredientes
    st.subheader("Conversa / Análise")
    if uploaded_image and option in ["Calcular Calorias do Prato", "Recomendar Receitas com Ingredientes"]:
        img = Image.open(uploaded_image)
        if option == "Calcular Calorias do Prato":
            imc = round(
                st.session_state.user.get("peso", 70) / (st.session_state.user.get("altura", 1.75) ** 2),
                2
            ) if st.session_state.user.get("altura") else 0
            analyze_dish_image(
                img, google_api_key,
                st.session_state.user.get("idade"),
                st.session_state.user.get("peso"),
                st.session_state.user.get("altura"),
                imc,
                st.session_state.user.get("nivel_atividade")
            )
        elif option == "Recomendar Receitas com Ingredientes":
            recommend_recipes_with_ingredients(img, google_api_key)

    # Fluxo de chat de texto simples
    if "messages" in st.session_state and menu_option == "Chat":
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                for content in msg["content"]:
                    if content["type"] == "text":
                        st.write(content["text"])
                    elif content["type"] == "image_url":
                        st.image(content["image_url"]["url"])

        if prompt := st.chat_input("Digite uma pergunta ou pedido de recomendação..."):
            st.session_state.messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
            add_message(st.session_state.chat_id, st.session_state.user["id"], "user", prompt)
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                st.write_stream(
                    stream_llm_response(
                        {"model": "gemini-2.0-flash", "temperature": 0.3},
                        google_api_key,
                        prompt_override=prompt
                    )
                )

if __name__ == "__main__":
    main()
