// Dashboard.js - Real-time updates for the Agentic AI Trader Dashboard

document.addEventListener('DOMContentLoaded', function() {
    // Initialize WebSocket connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname;
    // Use port 5678 for WebSocket as it runs on a separate server
    const wsURL = `${protocol}//${host}:5678`;
    let socket;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 10; // Increased for more attempts
    
    // Price data storage
    const priceData = {};
    
    // Connect to WebSocket server
    function connectWebSocket() {
        try {
            socket = new WebSocket(wsURL);
            
            // Connection event handlers
            socket.onopen = function() {
                console.log('WebSocket connection established');
                reconnectAttempts = 0;
                const statusElement = document.getElementById('websocket-status');
                if (statusElement) {
                    statusElement.classList.remove('status-stopped');
                    statusElement.classList.add('status-running');
                    statusElement.nextElementSibling.textContent = 'Live Updates: Connected';
                }
            };
            
            socket.onclose = function() {
                console.log('WebSocket connection closed');
                const statusElement = document.getElementById('websocket-status');
                if (statusElement) {
                    statusElement.classList.remove('status-running');
                    statusElement.classList.add('status-stopped');
                    statusElement.nextElementSibling.textContent = 'Live Updates: Disconnected';
                }
                
                // Attempt to reconnect with backoff
                if (reconnectAttempts < maxReconnectAttempts) {
                    reconnectAttempts++;
                    const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
                    console.log(`Attempting to reconnect in ${delay/1000} seconds...`);
                    setTimeout(connectWebSocket, delay);
                } else {
                    console.log('Max reconnect attempts reached. Please refresh the page.');
                    // Show alert to user
                    const alertElement = document.createElement('div');
                    alertElement.className = 'alert alert-danger';
                    alertElement.textContent = 'Connection lost. Please refresh the page to reconnect.';
                    const container = document.querySelector('.container');
                    if (container) {
                        container.prepend(alertElement);
                    }
                }
            };
            
            socket.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
            
            socket.onmessage = function(event) {
                try {
                    const message = JSON.parse(event.data);
                    
                    // Handle different message types
                    switch (message.type) {
                        case 'price_update':
                            handlePriceUpdate(message.data);
                            break;
                        case 'account_update':
                            handleAccountUpdate(message.data);
                            break;
                        case 'position_update':
                            handlePositionUpdate(message.data);
                            break;
                        case 'trade_executed':
                            handleTradeExecuted(message.data);
                            break;
                        case 'system_status':
                            handleSystemStatus(message.data);
                            break;
                        default:
                            console.log('Unknown message type:', message.type);
                    }
                } catch (e) {
                    console.error('Error handling message:', e, event.data);
                }
            };
        } catch (error) {
            console.error('Error creating WebSocket:', error);
            // Fallback to polling if WebSocket fails
            startPolling();
        }
    }
    
    // Fallback polling mechanism
    let pollingInterval;
    function startPolling() {
        console.log('Falling back to polling mechanism');
        // Clear any existing polling
        if (pollingInterval) clearInterval(pollingInterval);
        
        // Poll for data every 2 seconds
        pollingInterval = setInterval(function() {
            fetchPrices();
            fetchAccountInfo();
            fetchPositions();
        }, 2000);
    }
    
    // Fetch data via API
    function fetchPrices() {
        fetch('/api/prices')
            .then(response => response.json())
            .then(data => {
                if (data.prices) {
                    Object.entries(data.prices).forEach(([symbol, priceData]) => {
                        handlePriceUpdate(priceData);
                    });
                }
            })
            .catch(error => console.error('Error fetching prices:', error));
    }
    
    function fetchAccountInfo() {
        fetch('/api/account')
            .then(response => response.json())
            .then(data => {
                if (data.account) {
                    handleAccountUpdate(data.account);
                }
            })
            .catch(error => console.error('Error fetching account info:', error));
    }
    
    function fetchPositions() {
        fetch('/api/positions')
            .then(response => response.json())
            .then(data => {
                if (data.positions) {
                    // Create a full replacement update
                    handlePositionUpdate({
                        full_update: true,
                        positions: data.positions
                    });
                }
            })
            .catch(error => console.error('Error fetching positions:', error));
    }
    
    // Message handlers
    function handlePriceUpdate(data) {
        // Update price data store
        const symbol = data.symbol;
        priceData[symbol] = data;
        
        // Find the symbol information card
        const symbolInfoCard = document.querySelector('.card-header');
        if (!symbolInfoCard) return;
        
        const symbolCardHeader = Array.from(document.querySelectorAll('.card-header')).find(el => 
            el.textContent.includes('Symbol Information'));
        
        if (symbolCardHeader) {
            const headerText = symbolCardHeader.textContent.trim();
            const matches = headerText.match(/Symbol Information:\s*([A-Za-z0-9]+)/);
            const currentSymbol = matches ? matches[1].trim() : null;
            
            if (symbol === currentSymbol) {
                // Update symbol bid/ask display in the symbol information card
                const bidElement = document.querySelector('strong:contains("Bid")');
                const askElement = document.querySelector('strong:contains("Ask")');
                const spreadElement = document.querySelector('strong:contains("Spread")');
                
                if (bidElement) {
                    const bidValue = bidElement.nextElementSibling || bidElement.parentElement.lastChild;
                    if (bidValue) {
                        bidValue.textContent = data.bid;
                        bidValue.classList.add('price-flash');
                        setTimeout(() => bidValue.classList.remove('price-flash'), 500);
                    }
                }
                
                if (askElement) {
                    const askValue = askElement.nextElementSibling || askElement.parentElement.lastChild;
                    if (askValue) {
                        askValue.textContent = data.ask;
                        askValue.classList.add('price-flash');
                        setTimeout(() => askValue.classList.remove('price-flash'), 500);
                    }
                }
                
                if (spreadElement) {
                    const spreadValue = spreadElement.nextElementSibling || spreadElement.parentElement.lastChild;
                    if (spreadValue) {
                        spreadValue.textContent = `${data.spread} points`;
                    }
                }
            }
        }
        
        // Update any position with this symbol
        updatePositionPrices(symbol, data.bid, data.ask);
    }
    
    function handleAccountUpdate(data) {
        // Update account information display
        const accountInfo = document.querySelector('.card-header');
        if (!accountInfo) return;
        
        const accountInfoCard = Array.from(document.querySelectorAll('.card-header')).find(el => 
            el.textContent.includes('Account Information'));
            
        if (accountInfoCard) {
            const cardBody = accountInfoCard.nextElementSibling;
            if (!cardBody) return;
            
            const updateFields = [
                { label: 'Balance', value: `$${data.balance.toFixed(2)}` },
                { label: 'Equity', value: `$${data.equity.toFixed(2)}` },
                { label: 'Profit', value: `$${data.profit.toFixed(2)}`, class: data.profit >= 0 ? 'text-success' : 'text-danger' },
                { label: 'Margin', value: `$${data.margin.toFixed(2)}` },
                { label: 'Free Margin', value: `$${data.margin_free.toFixed(2)}` },
                { label: 'Margin Level', value: `${data.margin_level.toFixed(2)}%` },
                { label: 'Leverage', value: `${data.leverage}:1` }
            ];
            
            updateFields.forEach(field => {
                const row = Array.from(cardBody.querySelectorAll('.mb-2')).find(el => 
                    el.querySelector('strong') && el.querySelector('strong').textContent.includes(field.label));
                    
                if (row) {
                    const valueElement = row.querySelector('span') || row.childNodes[1];
                    if (valueElement) {
                        valueElement.textContent = field.value;
                        if (field.class) {
                            valueElement.className = field.class;
                        }
                    }
                }
            });
        }
    }
    
    function handlePositionUpdate(data) {
        const positionsHeader = Array.from(document.querySelectorAll('.card-header')).find(el => 
            el.textContent.includes('Open Positions'));
            
        if (!positionsHeader) return;
        
        const cardBody = positionsHeader.nextElementSibling;
        if (!cardBody) return;
        
        const positionsTable = cardBody.querySelector('table tbody');
        const noPositionsMessage = cardBody.querySelector('.alert');
        
        if (!positionsTable && !noPositionsMessage) return;
        
        // If this is a full update, clear and rebuild the entire positions table
        if (data.full_update) {
            if (data.positions.length === 0) {
                // Show no positions message
                if (positionsTable) {
                    positionsTable.parentElement.style.display = 'none';
                }
                if (!noPositionsMessage) {
                    const alertDiv = document.createElement('div');
                    alertDiv.className = 'alert alert-info';
                    alertDiv.innerHTML = '<p>No open positions.</p>';
                    cardBody.appendChild(alertDiv);
                } else {
                    noPositionsMessage.style.display = 'block';
                }
                return;
            }
            
            // We have positions, show the table
            if (positionsTable) {
                positionsTable.parentElement.style.display = 'table';
                positionsTable.innerHTML = '';
                
                data.positions.forEach(position => {
                    const row = createPositionRow(position);
                    positionsTable.appendChild(row);
                });
            }
            
            // Hide no positions message
            if (noPositionsMessage) {
                noPositionsMessage.style.display = 'none';
            }
        } else {
            // This is an update for a single position
            const position = data.position;
            const existingRow = positionsTable.querySelector(`tr[data-position-id="${position.ticket}"]`);
            
            if (existingRow) {
                // Update existing row
                const updatedRow = createPositionRow(position);
                positionsTable.replaceChild(updatedRow, existingRow);
            } else {
                // Add new row
                const newRow = createPositionRow(position);
                positionsTable.appendChild(newRow);
                
                // Hide no positions message if it exists
                if (noPositionsMessage) {
                    noPositionsMessage.style.display = 'none';
                }
                
                // Show table if it was hidden
                if (positionsTable) {
                    positionsTable.parentElement.style.display = 'table';
                }
            }
        }
    }
    
    function handleTradeExecuted(data) {
        // Show a notification
        showNotification('Trade Executed', `${data.type} ${data.volume} ${data.symbol} at ${data.price}`);
        
        // Refresh positions (could be done more efficiently by just updating the specific position)
        fetchPositions();
    }
    
    function handleSystemStatus(data) {
        // Update system status indicator
        const statusIndicator = document.querySelector('.status-indicator');
        if (!statusIndicator) return;
        
        const statusText = statusIndicator.nextElementSibling;
        
        if (data.is_running) {
            statusIndicator.classList.remove('status-stopped');
            statusIndicator.classList.add('status-running');
            if (statusText) statusText.textContent = 'Status: Running';
        } else {
            statusIndicator.classList.remove('status-running');
            statusIndicator.classList.add('status-stopped');
            if (statusText) statusText.textContent = 'Status: Stopped';
        }
        
        // Update the button
        const statusButton = document.querySelector('form[action^="/control/"] button');
        if (statusButton) {
            if (data.is_running) {
                statusButton.classList.remove('btn-success');
                statusButton.classList.add('btn-danger');
                statusButton.textContent = 'Stop System';
                statusButton.closest('form').action = '/control/stop';
            } else {
                statusButton.classList.remove('btn-danger');
                statusButton.classList.add('btn-success');
                statusButton.textContent = 'Start System';
                statusButton.closest('form').action = '/control/start';
            }
        }
    }
    
    // Helper functions
    function createPositionRow(position) {
        const row = document.createElement('tr');
        row.setAttribute('data-position-id', position.ticket);
        
        row.innerHTML = `
            <td>${position.ticket}</td>
            <td>${position.symbol}</td>
            <td>
                <span class="badge bg-${position.type === 0 ? 'success' : 'danger'}">
                    ${position.type === 0 ? 'BUY' : 'SELL'}
                </span>
            </td>
            <td>${position.volume}</td>
            <td>${position.price_open}</td>
            <td>${position.price_current}</td>
            <td class="${position.profit >= 0 ? 'text-success' : 'text-danger'}">
                $${position.profit.toFixed(2)}
            </td>
            <td>
                <form method="post" action="/close-position" class="d-inline">
                    <input type="hidden" name="position_id" value="${position.ticket}">
                    <button type="submit" class="btn btn-sm btn-danger">Close</button>
                </form>
            </td>
        `;
        
        return row;
    }
    
    function updatePositionPrices(symbol, bid, ask) {
        const positionsTable = document.querySelector('table tbody');
        if (!positionsTable) return;
        
        // Find all rows for this symbol
        positionsTable.querySelectorAll('tr').forEach(row => {
            const symbolCell = row.cells[1];
            if (symbolCell && symbolCell.textContent.trim() === symbol) {
                const typeCell = row.cells[2].querySelector('span');
                const currentPriceCell = row.cells[5];
                const profitCell = row.cells[6];
                
                if (!typeCell || !currentPriceCell) return;
                
                // Update current price based on position type
                const isBuy = typeCell.textContent.trim() === 'BUY';
                const price = isBuy ? bid : ask;
                const oldPrice = parseFloat(currentPriceCell.textContent);
                
                currentPriceCell.textContent = price;
                
                // Add visual indicator for price change
                if (price > oldPrice) {
                    currentPriceCell.classList.add('price-up');
                    setTimeout(() => currentPriceCell.classList.remove('price-up'), 1000);
                } else if (price < oldPrice) {
                    currentPriceCell.classList.add('price-down');
                    setTimeout(() => currentPriceCell.classList.remove('price-down'), 1000);
                }
            }
        });
    }
    
    function showNotification(title, message) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = 'toast';
        notification.setAttribute('role', 'alert');
        notification.setAttribute('aria-live', 'assertive');
        notification.setAttribute('aria-atomic', 'true');
        
        notification.innerHTML = `
            <div class="toast-header">
                <strong class="me-auto">${title}</strong>
                <small>just now</small>
                <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        `;
        
        // Add to the page
        let toastContainer = document.getElementById('toast-container');
        if (!toastContainer) {
            const container = document.createElement('div');
            container.id = 'toast-container';
            container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
            document.body.appendChild(container);
            toastContainer = container;
        }
        
        toastContainer.appendChild(notification);
        
        // Show the notification using Bootstrap's Toast
        const bootstrap = window.bootstrap;
        if (bootstrap && bootstrap.Toast) {
            const toast = new bootstrap.Toast(notification);
            toast.show();
            
            // Remove after it's hidden
            notification.addEventListener('hidden.bs.toast', function() {
                notification.remove();
            });
        } else {
            // Fallback if Bootstrap Toast is not available
            notification.style.display = 'block';
            setTimeout(() => {
                notification.remove();
            }, 5000);
        }
    }
    
    // Add CSS for price updates
    const style = document.createElement('style');
    style.textContent = `
        .price-flash {
            animation: flash-price 0.5s;
        }
        
        .price-up {
            color: #28a745 !important;
            transition: color 1s;
        }
        
        .price-down {
            color: #dc3545 !important;
            transition: color 1s;
        }
        
        @keyframes flash-price {
            0% { background-color: transparent; }
            50% { background-color: rgba(40, 167, 69, 0.3); }
            100% { background-color: transparent; }
        }
        
        #toast-container {
            z-index: 1050;
        }
    `;
    document.head.appendChild(style);
    
    // Create websocket status indicator
    const statusIndicator = document.querySelector('.status-indicator');
    if (statusIndicator) {
        const systemStatusContainer = statusIndicator.parentElement;
        const wsStatusContainer = document.createElement('div');
        wsStatusContainer.className = 'me-3';
        wsStatusContainer.innerHTML = `
            <span id="websocket-status" class="status-indicator status-stopped"></span>
            <span>Live Updates: Connecting...</span>
        `;
        systemStatusContainer.prepend(wsStatusContainer);
    }
    
    // Initialize
    connectWebSocket();
    
    // If websocket fails, poll every 3 seconds as fallback
    setTimeout(function() {
        if (!socket || socket.readyState !== WebSocket.OPEN) {
            startPolling();
        }
    }, 3000);
});
