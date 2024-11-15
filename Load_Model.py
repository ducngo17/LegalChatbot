import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate


model_path="vilm/vinallama-2.7b-chat"
token="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"


# Setup tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, token=token)


# Seting config
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, token=token)
config.init_device = "cuda"
config.temperature = 0.1
# config.max_length =300
# config.eos_token_id=tokenizer.eos_token_id
# config.pad_token_id=tokenizer.pad_token_id
# config.do_sample = True

bnb_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16,
                               )


model = AutoModelForCausalLM.from_pretrained(
    model_path,quantization_config=bnb_config,
    config=config,
    trust_remote_code=True , token=token
)


text_generation_pipeline = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=100,
)

my_pipeline = HuggingFacePipeline(pipeline=text_generation_pipeline)

#tao prompt template
template = """<|im_start|>system
Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""

prompt = PromptTemplate(template=template, input_variables=["question"])

#tao llm chain
llm_chain = LLMChain(prompt=prompt,
                     llm=my_pipeline
                     )

# question = "Hình tam giác có bao nhiêu cạnh?"

# result = llm_chain.invoke({"question":question})
# print(result)