from llama_index import GPTVectorStoreIndex, LLMPredictor, ServiceContext, Prompt
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Faiss Index Path
FAISS_INDEX = "vectorstore/"

# Custom prompt template
custom_prompt_template = """
[INST] <<SYS>>
You are a trained bot to guide people about Indian Law. You will answer user's query with your knowledge and the context provided. 
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
If you don't know the answer to a question, please don't share false information. 
Do not say thank you and tell you are an AI Assistant and be open about everything.
<</SYS>>
Context : {context}
Question : {question}
Answer : [/INST]
"""

# Return the custom prompt template
def set_custom_prompt_template():
    return custom_prompt_template

# Return the LLM with the enhanced Mistral 7B model
def load_llm():
    """
    Load the enhanced Mistral 7B model
    """
    # Model ID
    repo_id = 'mistralai/Mistral-7B-v0.1'

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(repo_id, device_map='auto', load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)

    # Create text generation pipeline
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, max_length=512)

    # Return LlamaIndex LLMPredictor with the Hugging Face pipeline
    llm_predictor = LLMPredictor(llm=pipe)
    return llm_predictor

# Return the chain
def qa_pipeline():
    """
    Create the QA pipeline
    """
    # Load the index from saved directory
    index = GPTVectorStoreIndex.load_from_disk(persist_dir=FAISS_INDEX)

    # Load the LLM
    llm = load_llm()

    # Set the custom prompt template
    prompt = Prompt(template=set_custom_prompt_template())

    # Create the service context (combines LLM and other configurations)
    service_context = ServiceContext.from_defaults(llm_predictor=llm, prompt_template=prompt)

    # Set the retriever with service context and prompt
    retriever = index.as_query_engine(service_context=service_context)

    return retriever
