-- Create Gumroad sales tracking table
CREATE TABLE IF NOT EXISTS gumroad_sales (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    sale_id VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) NOT NULL,
    customer_name VARCHAR(255),
    product_code VARCHAR(50),
    product_name VARCHAR(255),
    price DECIMAL(10,2),
    currency VARCHAR(10) DEFAULT 'USD',
    sale_timestamp TIMESTAMPTZ,
    convertkit_synced BOOLEAN DEFAULT false,
    stripe_synced BOOLEAN DEFAULT false,
    sendgrid_sent BOOLEAN DEFAULT false,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for faster lookups
CREATE INDEX IF NOT EXISTS idx_gumroad_sales_email ON gumroad_sales(email);
CREATE INDEX IF NOT EXISTS idx_gumroad_sales_product ON gumroad_sales(product_code);
CREATE INDEX IF NOT EXISTS idx_gumroad_sales_date ON gumroad_sales(sale_timestamp);

-- Create sales analytics view
CREATE OR REPLACE VIEW sales_analytics AS
SELECT
    DATE(sale_timestamp) as sale_date,
    COUNT(*) as daily_sales,
    SUM(price) as daily_revenue,
    COUNT(DISTINCT email) as unique_customers,
    ARRAY_AGG(DISTINCT product_code) as products_sold
FROM gumroad_sales
WHERE sale_timestamp > NOW() - INTERVAL '30 days'
GROUP BY DATE(sale_timestamp)
ORDER BY sale_date DESC;

-- Create product performance view
CREATE OR REPLACE VIEW product_performance AS
SELECT
    product_code,
    product_name,
    COUNT(*) as units_sold,
    SUM(price) as total_revenue,
    AVG(price) as avg_price,
    COUNT(DISTINCT email) as unique_buyers
FROM gumroad_sales
GROUP BY product_code, product_name
ORDER BY total_revenue DESC;

-- Add comment
COMMENT ON TABLE gumroad_sales IS 'Tracks all Gumroad sales for BrainOps products with integration status';