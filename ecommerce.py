"""Demo e-commerce catalog and order service backed by a local SQLite database."""
import sqlite3, os, secrets
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).resolve().parent / os.getenv('ECOMMERCE_DB', 'ecommerce.sqlite3')
CATALOG = [
 {'sku':'SKU1001','name':'Wireless Bluetooth Headphones','category':'Electronics','price':1499.0,'stock':25,'description':'Wireless over-ear headphones with microphone and long battery life.'},
 {'sku':'SKU1002','name':'USB-C Fast Charger 30W','category':'Electronics','price':699.0,'stock':40,'description':'Compact 30W USB-C fast charger.'},
 {'sku':'SKU1003','name':'Smart Fitness Watch','category':'Electronics','price':2499.0,'stock':12,'description':'Fitness watch with step tracking and heart-rate monitoring.'},
 {'sku':'SKU2001','name':'Everyday Cotton T-Shirt','category':'Fashion','price':499.0,'stock':35,'description':'Comfortable cotton crew-neck T-shirt.'},
 {'sku':'SKU2002','name':'Travel Backpack 25L','category':'Fashion','price':1199.0,'stock':18,'description':'Lightweight backpack with laptop sleeve.'},
 {'sku':'SKU3001','name':'Stainless Steel Water Bottle','category':'Home','price':349.0,'stock':50,'description':'Reusable insulated stainless-steel bottle.'},
 {'sku':'SKU3002','name':'Desk LED Lamp','category':'Home','price':899.0,'stock':16,'description':'Adjustable LED desk lamp with multiple brightness levels.'},
]
@contextmanager
def conn():
    c=sqlite3.connect(DB_PATH); c.row_factory=sqlite3.Row
    try: yield c; c.commit()
    finally: c.close()
def init_db():
    with conn() as c:
        c.execute('CREATE TABLE IF NOT EXISTS products (sku TEXT PRIMARY KEY,name TEXT,category TEXT,price REAL,stock INTEGER,description TEXT)')
        c.execute('CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY AUTOINCREMENT,order_ref TEXT UNIQUE,customer_name TEXT,customer_email TEXT,address TEXT,status TEXT,total REAL,items_json TEXT,created_at TEXT DEFAULT CURRENT_TIMESTAMP)')
        if c.execute('SELECT COUNT(*) FROM products').fetchone()[0]==0:
            c.executemany('INSERT INTO products VALUES (:sku,:name,:category,:price,:stock,:description)', CATALOG)
def list_products(query='', category='', limit=20):
    init_db(); q='%'+query.strip().lower()+'%'
    with conn() as c:
        rows=c.execute('SELECT * FROM products WHERE (lower(name) LIKE ? OR lower(description) LIKE ? OR lower(sku) LIKE ?) AND (? = "" OR lower(category)=lower(?)) ORDER BY name LIMIT ?', (q,q,q,category,category,max(1,min(limit,50)))).fetchall()
    return [dict(r) for r in rows]
def get_product(sku_or_name):
    rows=list_products(sku_or_name,limit=10)
    if not rows: return None
    exact=[r for r in rows if r['sku'].lower()==sku_or_name.lower() or r['name'].lower()==sku_or_name.lower()]
    return (exact or rows)[0]
def create_order(customer_name, customer_email, address, items, confirmed):
    if confirmed is not True: return {'ok':False,'error':'confirmation_required','message':'Read the cart, total, delivery address, and get explicit confirmation before placing the order.'}
    if not customer_name.strip() or not address.strip() or not items: return {'ok':False,'error':'missing_details','message':'Customer name, delivery address, and at least one item are required.'}
    init_db(); total=0; normalized=[]
    with conn() as c:
        for item in items:
            p=c.execute('SELECT * FROM products WHERE lower(sku)=lower(?) OR lower(name)=lower(?) LIMIT 1',(item.get('sku',''),item.get('name',''))).fetchone()
            qty=int(item.get('quantity',1))
            if not p or qty<1: return {'ok':False,'error':'invalid_item','message':f"Product not found or invalid quantity: {item.get('sku') or item.get('name')}"}
            if p['stock']<qty: return {'ok':False,'error':'out_of_stock','message':f"Only {p['stock']} units of {p['name']} are available."}
            total+=p['price']*qty; normalized.append({'sku':p['sku'],'name':p['name'],'quantity':qty,'unit_price':p['price'],'line_total':p['price']*qty})
        ref='ORD-'+secrets.token_hex(3).upper()
        import json
        c.execute('INSERT INTO orders(order_ref,customer_name,customer_email,address,status,total,items_json) VALUES(?,?,?,?,?,?,?)',(ref,customer_name,customer_email,address,'confirmed',total,json.dumps(normalized)))
        for item in normalized: c.execute('UPDATE products SET stock=stock-? WHERE sku=?',(item['quantity'],item['sku']))
    return {'ok':True,'order':{'order_ref':ref,'customer_name':customer_name,'address':address,'status':'confirmed','total':round(total,2),'items':normalized},'say':f'Your order is confirmed. Your order reference is {ref}. The total is ₹{total:.2f}.'}
def find_order(order_ref):
    import json
    init_db()
    with conn() as c: r=c.execute('SELECT * FROM orders WHERE upper(order_ref)=upper(?)',(order_ref.strip(),)).fetchone()
    if not r:return {'ok':False,'error':'not_found','message':'No order was found with that reference.'}
    return {'ok':True,'order':{'order_ref':r['order_ref'],'customer_name':r['customer_name'],'status':r['status'],'total':r['total'],'items':json.loads(r['items_json']),'created_at':r['created_at']}}
def list_orders(limit=50):
    import json
    init_db()
    with conn() as c: rows=c.execute('SELECT * FROM orders ORDER BY id DESC LIMIT ?', (max(1,min(limit,100)),)).fetchall()
    return [{'order_ref':r['order_ref'],'customer_name':r['customer_name'],'status':r['status'],'total':r['total'],'items':json.loads(r['items_json']),'created_at':r['created_at']} for r in rows]
