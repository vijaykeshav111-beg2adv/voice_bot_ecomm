# Deploying the Voice E-commerce Assistant

This project is a demo storefront and voice shopping assistant. Before internet deployment, use HTTPS (required for browser microphone access), configure a domain and reverse proxy, and set a strong `ADMIN_KEY`.

## Local deployment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Add your own API credentials to .env
python server.py
```
Browse to `http://localhost:8000`. The local SQLite database `ecommerce.sqlite3` is created when product/order APIs are first used. Back it up if you need to preserve demo orders.

## Production checklist
- Serve through HTTPS and a maintained reverse proxy.
- Set a long random `ADMIN_KEY`; add proper authentication and authorization to administrative order listing and MCP tools.
- Use a managed database and migrations instead of a local SQLite file for multi-worker production.
- Integrate a payment provider using its hosted/tokenized checkout; never collect card numbers, CVV, passwords, or OTPs in chat.
- Integrate a courier/shipping provider before advertising delivery estimates or live tracking.
- Add atomic inventory reservation, tax/shipping calculations, refunds/returns workflows, rate limiting, monitoring, backups, and privacy disclosures.
- Replace the sample catalog and store FAQ with approved, maintained business data.

## MCP
Run `python mcp_server.py` in a separate terminal. It uses stdio transport and should be launched by an MCP-compatible client. Do not expose the server or `list_recent_orders` to untrusted users without authentication/authorization.

## Demo limitations
This repository does not currently connect to a real payment gateway, courier, notification service, or automated returns/refunds system. Product prices and inventory are sample data; validate security and business rules before production use.
