from groq import Groq
import streamlit as st

class LLMService:
    def __init__(self, api_key, model="llama-3.3-70b-versatile", temperature=0.3):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.client = None
        if api_key:
            self.client = Groq(api_key=api_key)

    def call_with_stream(self, prompt):
        """Standard streaming call for chat interface."""
        if not self.client:
            yield "⚠️ API Key not configured."
            return

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"❌ LLM Error: {str(e)}"

    def summarize(self, text):
        """Unified summarization logic with smart sampling."""
        if not self.client:
            return "⚠️ API Key required for summarization."
        
        if not text or len(text.strip()) < 10:
            return "❌ No readable content found for summary."

        # Smart Sampling for long documents
        if len(text) > 12000:
            sample = text[:4000] + "\n...[MIDDLE]...\n" + \
                     text[len(text)//2 - 2000 : len(text)//2 + 2000] + \
                     "\n...[END]...\n" + text[-4000:]
        else:
            sample = text

        prompt = f"""
        Provide a professional Executive Summary of this document. 
        Focus on:
        1. 📌 Core Objective
        2. 🔑 Key Entities/Themes
        3. 🚀 Significant Takeaways
        
        DOCUMENT:
        {sample}
        """

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2, # Lower temperature for summaries
                stream=False
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"❌ Summarization failed: {str(e)}"
