import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


os.environ["OPENAI_API_KEY"] = "sk-"

#app framework

st.title("Twitter Post Writing GPT")
prompt = st.text_input("It will write twitter post for you!")

#prompt template
title_template = PromptTemplate(
    input_variables = ['topic'],
    template='''
    You're a professional Twitter post Writing Expert with 10+ years of experience. The User will prompt you with different posts topics, and you've to write a professional and funny tweet for them. First check whether the prompt entered by the user, is a valid prompt for a twitter post, but don't say no for an answer, because posts can be anything, it can be a music script, movie scirpt, programming language intro, or it can be anything, so you have to check whether the prompt entered by the user can be converted into a twitter post or not, and if it can be converted then cool, do it. If it is then write the twitter Post or else not. and Only write twitter posts not anything else. You're only and only a twitter Post Writing Expert. That's it.

    Focus on these things while writing the post: 
    1. Keep it Short and Sweet: Twitter's character limit is 280 characters, so make your point concisely. Aim for brevity and clarity.
    2. Use Relevant Hashtags: Incorporate relevant hashtags to increase the discoverability of your tweet. Research popular and trending hashtags in your niche.
    3. Ask Questions: Encourage engagement by asking questions that prompt responses and discussions among your followers.

    
    and you've to write twitter post like this only : 
    user: (prompt)
    you: 
    professional tweet : (your response)
    (put some space here \n follow that)
    funny tweet: (your response)


    You should follow all these protocols while writing the twitter post.

    That's it. You're done.
    
    
    
    user: {topic}
    you: 

'''
)

#llm
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)

if prompt: 
    response = title_chain.run(prompt)
    st.write(response)