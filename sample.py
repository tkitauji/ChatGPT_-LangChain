from gpt_index import GPTSimpleVectorIndex, SimpleWebPageReader, LLMPredictor, ServiceContext
from langchain.chat_models import ChatOpenAI


# LLM Predictor (gpt-3.5-turbo) + service context
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512)

# ドキュメントの読み込み
documents = SimpleWebPageReader().load_data(['https://ja.wikipedia.org/wiki/%E5%A4%A7%E8%B0%B7%E7%BF%94%E5%B9%B3'])

# indexの作成
# index = GPTSimpleVectorIndex.from_documents(documents)
index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

#index = GPTSimpleVectorIndex.from_documents(
#    documents = documents, service_context = service_context
#)

# indexの保存
index.save_to_disk("index.json")

# indexの照会
response = index.query("2023WBCの打撃成績")
print(response)
