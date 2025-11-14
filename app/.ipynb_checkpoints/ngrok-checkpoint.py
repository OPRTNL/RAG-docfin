from pyngrok import ngrok
import os
ngrok.set_auth_token(os.getenv("NGROK_AUTHTOKEN"))

# Start ngrok tunnel
public_url = ngrok.connect(addr='8501', proto='http', bind_tls=True)
print("Your Streamlit app is live at:", public_url)

# Wait for user input
input("Press Enter to stop the tunnel...\n")

# Kill ngrok
ngrok.kill()
print("ngrok tunnel closed.")