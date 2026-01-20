# CSV DATA ENTRY TEMPLATE

## Instructions for Adding Your 2000-2025 Export Data

This template shows the EXACT format needed. Copy your data from the Excel/PDF table and format it like this.

## Format Rules

1. **One row = one product in one year**
2. **Columns must be in this order:** year, product_code, product_name, quantity, quantity_unit, value_usd, source
3. **No commas in numbers** (wrong: 1,234,567 → correct: 1234567)
4. **Units must be:** Kg, No, M2, or L (case-sensitive)
5. **Product codes must match across years** (e.g., S.030205 for all Coco Peat entries)

## Example Rows (from your image)

```csv
year,product_code,product_name,quantity,quantity_unit,value_usd,source
2000,S.030205,Coco Peat Fiber Pith & Moulded products,75631947,Kg,16300003,EDB
2000,S.030107,Liquid Coconut Milk,0,Kg,0,EDB
2000,S.030301,Activated Carbon,16698937,Kg,17609082,EDB
2001,S.030205,Coco Peat Fiber Pith & Moulded products,61851855,Kg,13266000,EDB
2001,S.030107,Liquid Coconut Milk,0,Kg,0,EDB
2001,S.030301,Activated Carbon,16139809,Kg,15835000,EDB
```

## Full Template for One Product (Copy This Pattern)

```csv
year,product_code,product_name,quantity,quantity_unit,value_usd,source
2000,S.030205,Coco Peat Fiber Pith & Moulded products,75631947,Kg,16300003,EDB
2001,S.030205,Coco Peat Fiber Pith & Moulded products,61851855,Kg,13266000,EDB
2002,S.030205,Coco Peat Fiber Pith & Moulded products,68500000,Kg,15200000,EDB
2003,S.030205,Coco Peat Fiber Pith & Moulded products,75000000,Kg,18500000,EDB
2004,S.030205,Coco Peat Fiber Pith & Moulded products,82000000,Kg,22000000,EDB
2005,S.030205,Coco Peat Fiber Pith & Moulded products,90000000,Kg,26500000,EDB
2006,S.030205,Coco Peat Fiber Pith & Moulded products,<ADD_YOUR_DATA>,Kg,<ADD_YOUR_DATA>,EDB
2007,S.030205,Coco Peat Fiber Pith & Moulded products,<ADD_YOUR_DATA>,Kg,<ADD_YOUR_DATA>,EDB
...
2025,S.030205,Coco Peat Fiber Pith & Moulded products,<ADD_YOUR_DATA>,Kg,<ADD_YOUR_DATA>,EDB
```

## How to Convert Your Table to CSV

### Option 1: Excel/Google Sheets
1. Open your Excel file
2. Save As → CSV (Comma delimited)
3. Open in text editor
4. Verify format matches above
5. Replace `coconut_advisor/data/exports.csv`

### Option 2: Manual Entry
1. Copy the header row:
   ```
   year,product_code,product_name,quantity,quantity_unit,value_usd,source
   ```
2. For each product, add rows for years 2000-2025
3. Save as `exports.csv` (plain text)

### Option 3: Python Script (Quick)
```python
import pandas as pd

# If you have the data in Excel
df = pd.read_excel('your_data.xlsx')

# Rename columns to match
df = df.rename(columns={
    'Year': 'year',
    'Code': 'product_code',
    'Description': 'product_name',
    'Quantity': 'quantity',
    'Unit': 'quantity_unit',
    'Value': 'value_usd'
})

# Add source column
df['source'] = 'EDB'

# Remove zero-value rows (optional)
df = df[df['value_usd'] > 0]

# Save
df.to_csv('exports.csv', index=False)
```

## Common Mistakes to Avoid

❌ **Wrong:** Mixed units in quantity column
```csv
2000,S.030205,Coco Peat,75631947 Kg,Kg,16300003,EDB
```

✅ **Correct:** Number only in quantity, unit in separate column
```csv
2000,S.030205,Coco Peat Fiber Pith & Moulded products,75631947,Kg,16300003,EDB
```

---

❌ **Wrong:** Commas in numbers
```csv
2000,S.030205,Coco Peat,75,631,947,Kg,16,300,003,EDB
```

✅ **Correct:** No commas
```csv
2000,S.030205,Coco Peat Fiber Pith & Moulded products,75631947,Kg,16300003,EDB
```

---

❌ **Wrong:** Inconsistent product codes
```csv
2000,S.030205,Coco Peat,75631947,Kg,16300003,EDB
2001,S.30205,Coco Peat,61851855,Kg,13266000,EDB
```

✅ **Correct:** Same code every year
```csv
2000,S.030205,Coco Peat Fiber Pith & Moulded products,75631947,Kg,16300003,EDB
2001,S.030205,Coco Peat Fiber Pith & Moulded products,61851855,Kg,13266000,EDB
```

## Product Codes Reference (from your image)

```
S.030101 - Coconut Oil
S.030102 - Desiccated Coconut
S.030103 - Copra
S.030104 - Coconut Fresh Nuts
S.030105 - Coconut Milk Powder
S.030106 - Coconut Cream
S.030107 - Liquid Coconut Milk
S.030108 - Coconut Flour
S.030109 - Coconut Vinegar
S.030110 - Coconut Water
S.030111 - Poonac
S.030112 - Defatted Coconut
S.030199 - Other Coconut Kernel Products
S.030201 - Bristle Fiber
S.030202 - Mattress Fiber
S.030203 - Mixed Coir Fiber
S.030204 - Coir Yarn
S.030205 - Coco Peat Fiber Pith & Moulded products
S.030206 - Brooms & Brushes
S.030207 - Coir Carpets Mats Floor Coverings
S.030208 - Coconut Husk Chips
S.030209 - Coir Pads
S.030210 - Coir Twine & Ropes
S.030211 - Geo Textiles
S.030301 - Activated Carbon
S.030302 - Coconut Shell Pieces
S.030303 - Coconut Shell Powder
S.030304 - Coconut Shell Charcoal
S.030305 - Coconut Ekels
```

## Validation Checklist

Before replacing `exports.csv`, verify:

- [ ] First line is the header: `year,product_code,product_name,quantity,quantity_unit,value_usd,source`
- [ ] No blank lines between rows
- [ ] No commas inside values (except to separate columns)
- [ ] Each product has data for multiple years (at least 3-4)
- [ ] quantity_unit is one of: Kg, No, M2, L
- [ ] All numbers are positive (or zero)
- [ ] File is saved as `.csv` (not `.xlsx`)

## Quick Test

After adding your data:
```bash
cd coconut_advisor
python -c "import pandas as pd; df=pd.read_csv('data/exports.csv'); print(f'Rows: {len(df)}, Years: {df.year.min()}-{df.year.max()}, Products: {df.product_code.nunique()}')"
```

Expected output:
```
Rows: 650, Years: 2000-2025, Products: 28
```

If you see errors, check the format again!

## Need Help?

**Common error:** `ParserError: Error tokenizing data`
→ Check for extra commas or quotes in product names

**Common error:** `KeyError: 'year'`
→ Header row is missing or misspelled

**Common error:** `ValueError: could not convert string to float`
→ Commas in numbers or non-numeric values in quantity/value_usd columns

---

✅ Once your CSV is ready and validated, restart the server and you'll see analysis for all 26 years of data!
