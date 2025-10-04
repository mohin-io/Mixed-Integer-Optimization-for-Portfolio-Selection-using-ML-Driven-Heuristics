-- Portfolio Optimization Multi-Account Database Schema
-- PostgreSQL 15+
-- Version: 2.3.0

-- ============================================
-- USERS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(200),
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,

    CONSTRAINT username_length CHECK (LENGTH(username) >= 3),
    CONSTRAINT email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$')
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_active ON users(is_active);

-- ============================================
-- ACCOUNTS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS accounts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    account_name VARCHAR(100) NOT NULL,
    broker VARCHAR(50) NOT NULL,
    account_number VARCHAR(100),
    account_type VARCHAR(50) DEFAULT 'margin', -- margin, cash, ira, etc.
    initial_balance DECIMAL(15, 2) NOT NULL,
    current_balance DECIMAL(15, 2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    is_active BOOLEAN DEFAULT TRUE,
    is_paper BOOLEAN DEFAULT TRUE, -- paper trading vs live
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT positive_initial_balance CHECK (initial_balance >= 0),
    CONSTRAINT positive_current_balance CHECK (current_balance >= 0),
    CONSTRAINT unique_account_per_user UNIQUE(user_id, account_name)
);

CREATE INDEX idx_accounts_user ON accounts(user_id);
CREATE INDEX idx_accounts_broker ON accounts(broker);
CREATE INDEX idx_accounts_active ON accounts(is_active);

-- ============================================
-- PORTFOLIOS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS portfolios (
    id SERIAL PRIMARY KEY,
    account_id INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    portfolio_name VARCHAR(100) NOT NULL,
    strategy VARCHAR(50) NOT NULL, -- 'Max Sharpe', 'Min Variance', etc.
    risk_tolerance DECIMAL(3, 2) DEFAULT 0.50,
    max_position_size DECIMAL(3, 2) DEFAULT 0.10, -- 10% max per position
    rebalance_frequency VARCHAR(20) DEFAULT 'monthly', -- daily, weekly, monthly, quarterly
    auto_rebalance BOOLEAN DEFAULT FALSE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_risk_tolerance CHECK (risk_tolerance BETWEEN 0 AND 1),
    CONSTRAINT valid_max_position CHECK (max_position_size BETWEEN 0 AND 1),
    CONSTRAINT unique_portfolio_per_account UNIQUE(account_id, portfolio_name)
);

CREATE INDEX idx_portfolios_account ON portfolios(account_id);
CREATE INDEX idx_portfolios_strategy ON portfolios(strategy);
CREATE INDEX idx_portfolios_auto_rebalance ON portfolios(auto_rebalance);

-- ============================================
-- POSITIONS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    asset_class VARCHAR(50) DEFAULT 'equity', -- equity, bond, commodity, crypto, etc.
    quantity DECIMAL(15, 4) NOT NULL,
    avg_cost DECIMAL(15, 4) NOT NULL,
    current_price DECIMAL(15, 4),
    market_value DECIMAL(15, 2),
    unrealized_pnl DECIMAL(15, 2),
    realized_pnl DECIMAL(15, 2) DEFAULT 0,
    weight DECIMAL(5, 4), -- position weight in portfolio
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_position_per_portfolio UNIQUE(portfolio_id, symbol)
);

CREATE INDEX idx_positions_portfolio ON positions(portfolio_id);
CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_positions_asset_class ON positions(asset_class);

-- ============================================
-- TRANSACTIONS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    transaction_type VARCHAR(10) NOT NULL, -- BUY, SELL, DIVIDEND, FEE
    quantity DECIMAL(15, 4) NOT NULL,
    price DECIMAL(15, 4) NOT NULL,
    total_amount DECIMAL(15, 2) NOT NULL,
    commission DECIMAL(10, 2) DEFAULT 0,
    fees DECIMAL(10, 2) DEFAULT 0,
    transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    order_id VARCHAR(100),
    notes TEXT,

    CONSTRAINT valid_transaction_type CHECK (transaction_type IN ('BUY', 'SELL', 'DIVIDEND', 'FEE', 'SPLIT'))
);

CREATE INDEX idx_transactions_portfolio ON transactions(portfolio_id);
CREATE INDEX idx_transactions_symbol ON transactions(symbol);
CREATE INDEX idx_transactions_date ON transactions(transaction_date);
CREATE INDEX idx_transactions_type ON transactions(transaction_type);

-- ============================================
-- PERFORMANCE HISTORY TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS performance_history (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    total_value DECIMAL(15, 2) NOT NULL,
    cash_value DECIMAL(15, 2),
    positions_value DECIMAL(15, 2),
    daily_return DECIMAL(10, 6),
    cumulative_return DECIMAL(10, 6),
    sharpe_ratio DECIMAL(10, 4),
    sortino_ratio DECIMAL(10, 4),
    volatility DECIMAL(10, 6),
    max_drawdown DECIMAL(10, 6),
    num_positions INTEGER,

    CONSTRAINT unique_portfolio_date UNIQUE(portfolio_id, date)
);

CREATE INDEX idx_performance_portfolio ON performance_history(portfolio_id);
CREATE INDEX idx_performance_date ON performance_history(date);
CREATE INDEX idx_performance_portfolio_date ON performance_history(portfolio_id, date);

-- ============================================
-- REBALANCE HISTORY TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS rebalance_history (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    rebalance_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    strategy_used VARCHAR(50),
    old_weights JSONB, -- Store old weights as JSON
    new_weights JSONB, -- Store new weights as JSON
    trades_executed INTEGER DEFAULT 0,
    total_cost DECIMAL(10, 2) DEFAULT 0, -- commissions + fees
    success BOOLEAN DEFAULT TRUE,
    notes TEXT
);

CREATE INDEX idx_rebalance_portfolio ON rebalance_history(portfolio_id);
CREATE INDEX idx_rebalance_date ON rebalance_history(rebalance_date);

-- ============================================
-- WATCHLISTS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS watchlists (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    watchlist_name VARCHAR(100) NOT NULL,
    symbols TEXT[], -- Array of ticker symbols
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_watchlist_per_user UNIQUE(user_id, watchlist_name)
);

CREATE INDEX idx_watchlists_user ON watchlists(user_id);

-- ============================================
-- ALERTS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolios(id) ON DELETE CASCADE,
    alert_type VARCHAR(50) NOT NULL, -- drawdown, sharpe, volatility, etc.
    severity VARCHAR(20) NOT NULL, -- info, warning, critical
    message TEXT NOT NULL,
    metrics JSONB, -- Store alert metrics as JSON
    triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMP,
    acknowledged_by INTEGER REFERENCES users(id)
);

CREATE INDEX idx_alerts_portfolio ON alerts(portfolio_id);
CREATE INDEX idx_alerts_type ON alerts(alert_type);
CREATE INDEX idx_alerts_severity ON alerts(severity);
CREATE INDEX idx_alerts_triggered ON alerts(triggered_at);

-- ============================================
-- API KEYS TABLE (for secure broker credentials)
-- ============================================
CREATE TABLE IF NOT EXISTS api_keys (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    broker VARCHAR(50) NOT NULL,
    key_name VARCHAR(100) NOT NULL,
    api_key_encrypted TEXT NOT NULL,
    api_secret_encrypted TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP,

    CONSTRAINT unique_key_per_user UNIQUE(user_id, broker, key_name)
);

CREATE INDEX idx_api_keys_user ON api_keys(user_id);
CREATE INDEX idx_api_keys_broker ON api_keys(broker);

-- ============================================
-- AUDIT LOG TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL, -- login, logout, create_portfolio, place_order, etc.
    entity_type VARCHAR(50), -- user, account, portfolio, position, etc.
    entity_id INTEGER,
    details JSONB,
    ip_address VARCHAR(45),
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_user ON audit_log(user_id);
CREATE INDEX idx_audit_action ON audit_log(action);
CREATE INDEX idx_audit_date ON audit_log(created_at);

-- ============================================
-- FUNCTIONS AND TRIGGERS
-- ============================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_accounts_updated_at BEFORE UPDATE ON accounts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_portfolios_updated_at BEFORE UPDATE ON portfolios
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_watchlists_updated_at BEFORE UPDATE ON watchlists
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to calculate portfolio metrics
CREATE OR REPLACE FUNCTION calculate_position_metrics()
RETURNS TRIGGER AS $$
BEGIN
    -- Calculate market value
    NEW.market_value = NEW.quantity * COALESCE(NEW.current_price, NEW.avg_cost);

    -- Calculate unrealized P&L
    NEW.unrealized_pnl = NEW.quantity * (COALESCE(NEW.current_price, NEW.avg_cost) - NEW.avg_cost);

    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER calculate_position_metrics_trigger
    BEFORE INSERT OR UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION calculate_position_metrics();

-- ============================================
-- VIEWS FOR COMMON QUERIES
-- ============================================

-- Portfolio summary view
CREATE OR REPLACE VIEW portfolio_summary AS
SELECT
    p.id as portfolio_id,
    p.portfolio_name,
    p.strategy,
    a.account_name,
    u.username,
    COUNT(pos.id) as num_positions,
    COALESCE(SUM(pos.market_value), 0) as total_value,
    COALESCE(SUM(pos.unrealized_pnl), 0) as total_unrealized_pnl,
    MAX(pos.last_updated) as last_updated
FROM portfolios p
JOIN accounts a ON p.account_id = a.id
JOIN users u ON a.user_id = u.id
LEFT JOIN positions pos ON p.id = pos.portfolio_id
GROUP BY p.id, p.portfolio_name, p.strategy, a.account_name, u.username;

-- User account summary view
CREATE OR REPLACE VIEW user_account_summary AS
SELECT
    u.id as user_id,
    u.username,
    u.email,
    COUNT(DISTINCT a.id) as num_accounts,
    COUNT(DISTINCT p.id) as num_portfolios,
    SUM(a.current_balance) as total_balance
FROM users u
LEFT JOIN accounts a ON u.id = a.user_id
LEFT JOIN portfolios p ON a.id = p.account_id
GROUP BY u.id, u.username, u.email;

-- Recent transactions view
CREATE OR REPLACE VIEW recent_transactions AS
SELECT
    t.id,
    t.transaction_date,
    t.transaction_type,
    t.symbol,
    t.quantity,
    t.price,
    t.total_amount,
    p.portfolio_name,
    a.account_name,
    u.username
FROM transactions t
JOIN portfolios p ON t.portfolio_id = p.id
JOIN accounts a ON p.account_id = a.id
JOIN users u ON a.user_id = u.id
ORDER BY t.transaction_date DESC;

-- ============================================
-- SAMPLE DATA (for development/testing)
-- ============================================

-- Insert default admin user (password: 'admin123' - change in production!)
-- Password hash generated with bcrypt
INSERT INTO users (username, email, password_hash, full_name, is_admin)
VALUES ('admin', 'admin@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYzpLHJ/SN2', 'System Administrator', TRUE)
ON CONFLICT (username) DO NOTHING;

-- Insert demo user
INSERT INTO users (username, email, password_hash, full_name)
VALUES ('demo_user', 'demo@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYzpLHJ/SN2', 'Demo User')
ON CONFLICT (username) DO NOTHING;

-- ============================================
-- GRANTS (adjust as needed for your setup)
-- ============================================

-- Grant permissions to application user
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO portfolio_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO portfolio_app;

-- ============================================
-- COMMENTS (for documentation)
-- ============================================

COMMENT ON TABLE users IS 'User accounts for the portfolio management system';
COMMENT ON TABLE accounts IS 'Trading accounts linked to users (can have multiple per user)';
COMMENT ON TABLE portfolios IS 'Portfolios within accounts (multiple strategies per account)';
COMMENT ON TABLE positions IS 'Current holdings in each portfolio';
COMMENT ON TABLE transactions IS 'Historical transaction records';
COMMENT ON TABLE performance_history IS 'Daily performance snapshots for portfolios';
COMMENT ON TABLE rebalance_history IS 'Portfolio rebalancing records';
COMMENT ON TABLE alerts IS 'System alerts and notifications';
COMMENT ON TABLE audit_log IS 'Audit trail of all user actions';

-- End of schema
