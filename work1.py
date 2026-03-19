import pandas as pd
import nest_asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import uvicorn
import os                       # Для getenv, чтобы спрятать ключ
from dotenv import load_dotenv  # Библиотека для чтения файлов .env

# Ищем файл .env в папке со скриптом и загружаем переменные из него в память
load_dotenv() 
# Записываем ключ в переменную
SECRET_KEY = os.getenv("GEMINI_API_KEY")
# Для ошибки
if not SECRET_KEY:
    raise ValueError("Ключ API не найден! Проверь наличие файла .env и его содержимое.")
nest_asyncio.apply()

app = FastAPI(title="Аналитика Конкурентов WB API")
client = AsyncOpenAI(
    api_key=SECRET_KEY, # Загружаем ключик
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Загружаем базу
df = pd.read_csv("data.csv", sep=";") 
df['Latest rating avg'] = df['Latest rating avg'].astype(str).str.replace(',', '.').astype(float)
print(f"✅ База загружена! Товаров: {len(df)}")


# --- СХЕМЫ (Models) ---
class ProductInput(BaseModel): # Входящий товар
    price: int
    rating: float
    comments: int
    sales: int 
    balance: int
    temperature: float = 0

class Advantage(BaseModel): # Преимущество
    category: str = Field(alias="категория")
    description: str = Field(alias="описание")
    significance: str = Field(alias="значимость")
    how_to_use: str = Field(alias="как_использовать")

class Problem(BaseModel): # Проблема
    category: str = Field(alias="категория")
    description: str = Field(alias="описание")
    criticality: str = Field(alias="критичность")
    how_to_fix: str = Field(alias="как_исправить")

class MarketAnalysis(BaseModel): # Анализ Рынка
    general_conclusion: str = Field(alias="общий_вывод")
    strengths: list[Advantage] = Field(alias="сильные_стороны")
    weaknesses: list[Problem] = Field(alias="слабые_места")
    action_plan: list[str] = Field(alias="план_действий")
    recommended_price: int = Field(alias="рекомендуемая_цена")
    price_justification: str = Field(alias="обоснование_цены")


# Эндпоинт 
@app.post("/analyze", response_model=MarketAnalysis)
async def analyze_competitors(product: ProductInput):
    # Берем ТОП-50 по выручке
    top_50_df = df.sort_values(by='Revenue', ascending=False).head(50)
    
    # Считаем среднюю цену и медианы для остального
    market_avg_price = int(top_50_df['Price with WB wallet'].mean()) 
    market_avg_rating = round(top_50_df['Latest rating avg'].mean(), 2)
    market_median_comments = int(top_50_df['Comments'].median())
    
    market_median_sales = int(top_50_df['Sales'].median()) if 'Sales' in top_50_df.columns else 0
    market_median_balance = int(top_50_df['Balance'].median()) if 'Balance' in top_50_df.columns else 0

    # Предрасчет оборачиваемости
    if product.sales > 0:
        months_of_stock = round(product.balance / product.sales, 1)
    else:
        months_of_stock = 99.0
        
    # Пишем промт
    system_prompt = """Ты Senior-Аналитик маркетплейса Wildberries. Твоя задача — дать бизнесу мощные, рабочие рекомендации.
    Строго соблюдай правила:
    1. Математика: Если наша цена МЕНЬШЕ медианы, мы продаем дешевле рынка.
    2. Оборачиваемость: В данных ниже указано, на сколько месяцев хватит наших запасов.
    Если запасов МЕНЕЕ чем на 1 месяц (например, 0.4) - пиши, что это риск Out-of-Stock, нужно СРОЧНО делать поставку и поднять цену для замедления продаж.
    Если запасов БОЛЬШЕ чем на 2 месяца - это заморозка капитала, предложи акции для слива стока.
    3. Анализ ВСЕХ метрик: ОБЯЗАТЕЛЬНО проанализируй ВСЕ 5 метрик (Цена, Рейтинг, Отзывы, Продажи, Остатки).
    Хорошие показатели заноси в 'сильные_стороны', плохие - в 'слабые_места'.

    ОБЯЗАТЕЛЬНО верни СТРОГИЙ JSON. Никакого текста до или после JSON.
    Используй ТОЛЬКО следующие ключи на русском языке:
    {
      "общий_вывод": "Текст",
      "сильные_стороны": [
        {
          "категория": "Текст",
          "описание": "Текст",
          "значимость": "Высокая / Средняя / Низкая",
          "как_использовать": "Текст"
        }
      ],
      "слабые_места": [
        {
          "категория": "Текст",
          "описание": "Текст",
          "критичность": "Высокая / Средняя / Низкая",
          "как_исправить": "Текст"
        }
      ],
      "план_действий": ["Шаг 1", "Шаг 2"],
      "рекомендуемая_цена": 1000,
      "обоснование_цены": "Текст"
    }"""
    
    user_prompt = f"""
    Сравни метрики:
    1. ЦЕНА: Наша = {product.price} руб. | У конкурентов = {market_avg_price} руб.
    2. РЕЙТИНГ: Наш = {product.rating} | У конкурентов = {market_avg_rating}
    3. ОТЗЫВЫ: Наши = {product.comments} | У конкурентов = {market_median_comments}
    4. ПРОДАЖИ (шт): Наши = {product.sales} | У конкурентов = {market_median_sales}
    5. ОСТАТКИ (шт): Наши = {product.balance} | У конкурентов = {market_median_balance}

    ВАЖНЫЙ РАСЧЕТ ОБОРАЧИВАЕМОСТИ:
    Наших запасов на складе при текущих продажах хватит ровно на {months_of_stock} месяцев. Опирайся на эту цифру для выводов по остаткам!
    """
    
    try:
        response = await client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=product.temperature,
            response_format={ "type": "json_object" } # <--- Тот самый механизм из задания!
        )
        
        # возвращаем чистый JSON
        clean_json = response.choices[0].message.content.strip()
        
        # Pydantic сам парсит JSON и проверяет ключи
        return MarketAnalysis.model_validate_json(clean_json)
        
    except Exception as e:
        print(f"Ошибка API: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    config = uvicorn.Config(app, host="127.0.0.1", port=8009)
    server = uvicorn.Server(config)
    await server.serve()