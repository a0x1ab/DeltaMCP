# AutoMCP - OpenAPI to MCP Generator

Download the executable and run it directly.

## 🚀 Quick Start

1. Download `automcp.exe` from this repository
2. Run: `./automcp.exe --input your_spec.yaml --output output_dir`
3. Use the generated MCP server with Claude or Cursor

## 📖 Usage

```bash
# Basic usage
./automcp.exe --input spec.yaml --output output_dir

# Advanced usage
./automcp.exe --input spotify.yaml --output spotify_mcp --include-tags "playlists,albums"
```

## 🚀 Running the Generated Server

### 1. Set up authentication
Edit the generated `.env` file with your credentials:
```bash
cd output_directory
# Edit .env file with your API credentials
```

### 2. For OAuth2 services, start the login server first:
```bash
python oauth_login_server.py
```

**Available OAuth2 endpoints:**
- **Authorization Code Flow**: Visit `http://localhost:8888/login` to authenticate
- **Implicit Flow**: Visit `http://localhost:8888/login_implicit` to authenticate  
- **Client Credentials Flow**: Visit `http://localhost:8888/token_client_credentials` to get access token
- **PKCE Support**: Add `PKCE=true` to your `.env` file for enhanced security

**Note**: The specific endpoints available depend on the OAuth2 flows defined in the OpenAPI specification.

### 3. Run the MCP server:
```bash
python server_stub.py
```

## 🎧 Example Workflow (Spotify)

### Spotify API Example

1. **Generate the server stub:**
```bash
python main.py --input spotify.yaml --output spotify_mcp
```

2. **Configure authentication:**
```bash
cd spotify_mcp
# Edit .env with your Spotify API credentials
```

3. **Start OAuth2 login (if needed):**
```bash
python oauth_login_server.py
# Visit http://localhost:8888/login
```

4. **Run the MCP server:**
```bash
python server_stub.py
```

## ⚙️ Tool Integration: Claude Desktop vs Cursor

### ✅ Claude Integration (Claude Desktop Only)
You must have the Claude Desktop App installed. Then:

1. Go to **Settings → Developer Settings**
2. Click **“Edit Configuration”** (`claude_desktop_config.json`)
3. Paste the following into your config (example shown for a Spotify tool):

```json
{
  "mcpServers": {
    "spotify": {
      "command": "python",                               
      "args": [
        "C:/Path/To/Your/AutoMCP/spotify_mcp/server_stub.py",             // 🔁 Change to the full path of your tool folder                          
      ]
    }
  }
}
```

---

### ✅ Cursor Integration
You can use the same config block, but the file that needs to be updated is `mcp.json` inside Cursor.

## 🛡️ Error Handling

Robust error handling is built into AutoMCP, but issues may still arise if your OpenAPI specification is malformed or incomplete. Common sources of errors include:

- **Missing or Incorrect Authentication Schemes:**  
  Ensure all security schemes (API keys, OAuth2, etc.) are properly defined in your spec.
- **Parameter Type Mismatches:**  
  Check that parameter types in your spec (e.g., string, integer, boolean) match their intended usage.
- **Invalid or Missing Base URLs:**  
  Verify that the `servers` section (OpenAPI 3.x) or `host`/`basePath` (Swagger 2.0) is correctly set.
  
Review error messages when calling the MCP server tools for details on what went wrong.
**Happy MCP Server Generation! 🚀**

## 📚 Citation

If you use AutoMCP in your research or development, please consider citing our paper:

```bibtex
@misc{mastouri2025makingrestapisagentready,
  title     = {Making REST APIs Agent-Ready: From OpenAPI to Model Context Protocol Servers for Tool-Augmented LLMs},
  author    = {Meriem Mastouri and Emna Ksontini and Wael Kessentini},
  year      = {2025},
  eprint    = {2507.16044},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SE},
  url       = {https://arxiv.org/abs/2507.16044}
}




