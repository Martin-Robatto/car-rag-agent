import gradio as gr
from car_agent import create_car_agent

# Create the agent
agent = create_car_agent()

def chat_with_agent(message, history):
    # Run the agent on the user's message
    response = agent.run(message)
    return response

# Set up the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🚗 Car Information Agent")
    gr.Markdown("Ask me anything about cars from the classic auto dataset. I can help you find cars with specific features, compare models, or make recommendations.")
    
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label="Ask about cars")
    clear = gr.Button("Clear")
    
    def respond(message, chat_history):
        bot_message = chat_with_agent(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(share=True)  # Set share=False if you don't want a public link 