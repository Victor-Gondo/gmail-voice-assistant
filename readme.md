pip install -r requirements.txt

ensure your gmail api access is turned on in google cloud console

place your gcp-oauth.keys.json in your directory or in the "rootfolder"/".gmail-mcp"

run this in terminal to install and authorize:
npx -y @smithery/cli install @gongrzhe/server-gmail-autoauth-mcp
npx @gongrzhe/server-gmail-autoauth-mcp auth

run agent.py.

place your gcp-oauth.keys.json in your directory or in the root folder ".gmail-mcp"