// Main JavaScript file for Fyona Canvas Editor - Click-to-Select Implementation

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const canvas = document.getElementById('canvas');
    const addTextBtn = document.getElementById('add-text');
    const addImageBtn = document.getElementById('add-image');
    const saveLayoutBtn = document.getElementById('save-layout');
    const loadLayoutBtn = document.getElementById('load-layout');
    const clearCanvasBtn = document.getElementById('clear-canvas');
    const zoomInBtn = document.getElementById('zoom-in');
    const zoomOutBtn = document.getElementById('zoom-out');
    const zoomLevelDisplay = document.getElementById('zoom-level');
    const blockPropertiesPanel = document.getElementById('block-properties');
    const canvasPropertiesPanel = document.getElementById('canvas-properties');
    const blockContentInput = document.getElementById('block-content');
    const blockXInput = document.getElementById('block-x');
    const blockYInput = document.getElementById('block-y');
    const blockWidthInput = document.getElementById('block-width');
    const blockHeightInput = document.getElementById('block-height');
    const blockBgColorInput = document.getElementById('block-bg-color');
    const deleteBlockBtn = document.getElementById('delete-block');
    
    // State
    let selectedBlock = null;
    let isDragging = false;
    let isResizing = false;
    let dragStartX, dragStartY;
    let originalX, originalY;
    let originalWidth, originalHeight;
    let zoomLevel = 1;
    let zoomStep = 0.1;
    const MIN_BLOCK_WIDTH = 50;
    const MIN_BLOCK_HEIGHT = 30;
    const DEFAULT_TEXT_DIMENSIONS = { width: 240, height: 120 };
    const DEFAULT_IMAGE_DIMENSIONS = { width: 200, height: 160 };

    function toNumber(value, fallback = 0) {
        if (typeof value === 'number') {
            return Number.isFinite(value) ? value : fallback;
        }
        if (typeof value === 'string') {
            const trimmed = value.trim();
            if (!trimmed) return fallback;
            const parsed = parseFloat(trimmed);
            return Number.isFinite(parsed) ? parsed : fallback;
        }
        return fallback;
    }

    function getNumericStyleValue(element, property, fallback = 0) {
        if (!element) return fallback;
        const inlineValue = toNumber(element.style[property], NaN);
        if (Number.isFinite(inlineValue)) {
            return inlineValue;
        }
        const computedStyle = window.getComputedStyle(element);
        return toNumber(computedStyle[property], fallback);
    }
    
    // Initialize the application
    function initApp() {
        setupEventListeners();
        loadLayout();
        updateZoomDisplay();
    }
    
    // Set up event listeners
    function setupEventListeners() {
        // Toolbar buttons
        addTextBtn.addEventListener('click', () => addBlock('text'));
        addImageBtn.addEventListener('click', () => addBlock('image'));
        saveLayoutBtn.addEventListener('click', saveLayout);
        loadLayoutBtn.addEventListener('click', loadLayout);
        clearCanvasBtn.addEventListener('click', clearCanvas);
        
        // Zoom controls
        zoomInBtn.addEventListener('click', zoomIn);
        zoomOutBtn.addEventListener('click', zoomOut);
        
        // Canvas events
        canvas.addEventListener('click', handleCanvasClick);
        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
        
        // Property panel events
        blockContentInput.addEventListener('input', updateBlockProperties);
        blockXInput.addEventListener('input', updateBlockProperties);
        blockYInput.addEventListener('input', updateBlockProperties);
        blockWidthInput.addEventListener('input', updateBlockProperties);
        blockHeightInput.addEventListener('input', updateBlockProperties);
        blockBgColorInput.addEventListener('input', updateBlockProperties);
        deleteBlockBtn.addEventListener('click', deleteSelectedBlock);
    }
    
    // Add a new block to the canvas with transparent background and black text
    async function addBlock(type) {
        // Default to transparent background with black text
        const backgroundColor = 'rgba(255, 255, 255, 0.0)'; // Transparent background
        const textColor = '#000000'; // Black text
            
        const blockData = {
            type: type,
            content: type === 'text' ? 'Sample Text' : '[Image Block]',
            position: {
                left: getRandomPosition(100, 300),
                top: getRandomPosition(100, 300),
                width: type === 'text' ? getRandomSize(180, 250) : getRandomSize(150, 200),
                height: type === 'text' ? getRandomSize(80, 120) : getRandomSize(120, 180)
            },
            backgroundColor: backgroundColor,
            textColor: textColor,
            borderRadius: '12px'
        };
        
        try {
            const response = await fetch('/api/block', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    operation: 'add',
                    block: blockData
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                renderBlock(data.block);
            } else {
                console.error('Failed to add block:', data.error);
            }
        } catch (error) {
            console.error('Error adding block:', error);
            // Fallback to client-side creation
            renderBlock({
                id: 'block-' + Date.now(),
                ...blockData
            });
        }
    }
    
    // Render a block on the canvas with transparent background and black text
    function renderBlock(blockData) {
        const block = document.createElement('div');
        block.className = `block ${blockData.type}`;
        block.id = blockData.id;

        const position = blockData.position || {};
        const dimensionDefaults = blockData.type === 'image' ? DEFAULT_IMAGE_DIMENSIONS : DEFAULT_TEXT_DIMENSIONS;
        const left = toNumber(position.left ?? position.x, 0);
        const top = toNumber(position.top ?? position.y, 0);
        const width = Math.max(MIN_BLOCK_WIDTH, toNumber(position.width ?? position.w, dimensionDefaults.width));
        const height = Math.max(MIN_BLOCK_HEIGHT, toNumber(position.height ?? position.h, dimensionDefaults.height));

        block.style.left = left + 'px';
        block.style.top = top + 'px';
        block.style.width = width + 'px';
        block.style.height = height + 'px';
        blockData.position = { ...position, left, top, width, height };
        
        // Apply transparent background styling
        if (blockData.backgroundColor) {
            block.style.background = blockData.backgroundColor;
        } else {
            block.style.background = 'rgba(255, 255, 255, 0.0)'; // Default transparent
        }
        
        if (blockData.borderRadius) {
            block.style.borderRadius = blockData.borderRadius;
        }
        
        // Add subtle animation
        block.style.animation = 'fadeIn 0.3s ease-out';
        
        // Create block content container
        const blockContent = document.createElement('div');
        blockContent.className = 'block-content';
        blockContent.contentEditable = false; // Initially not editable
        blockContent.innerHTML = blockData.content;
        
        // Apply text color
        if (blockData.textColor) {
            blockContent.style.color = blockData.textColor;
        } else {
            blockContent.style.color = '#000000'; // Default black text
        }
        
        // Style content area
        blockContent.style.borderRadius = blockData.borderRadius ? `calc(${blockData.borderRadius} - 4px)` : '8px';
        
        block.appendChild(blockContent);
        
        // Add resize handle
        const resizeHandle = document.createElement('div');
        resizeHandle.className = 'resize-handle';
        block.appendChild(resizeHandle);
        
        canvas.appendChild(block);
        
        // Add event listeners to the new block
        block.addEventListener('click', (e) => {
            // Only select if we're not currently dragging
            if (!isDragging && !isResizing) {
                e.stopPropagation();
                selectAndEditBlock(block);
            }
        });
        
        // Add hover effect
        block.addEventListener('mouseenter', () => {
            block.style.transform = 'translateY(-2px)';
            block.style.boxShadow = '0 8px 28px rgba(0, 0, 0, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.3)';
            // Slightly lighten the background on hover
            if (blockData.backgroundColor && blockData.backgroundColor.includes('rgba')) {
                const currentOpacity = parseFloat(blockData.backgroundColor.split(',')[3]);
                if (currentOpacity < 0.3) {
                    block.style.background = blockData.backgroundColor.replace(/[\d.]+\)/, '0.1)');
                }
            }
        });
        
        block.addEventListener('mouseleave', () => {
            if (block !== selectedBlock) {
                block.style.transform = 'translateY(0)';
                block.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.2)';
                block.style.background = blockData.backgroundColor;
            }
        });
        
        // Add resize functionality
        resizeHandle.addEventListener('mousedown', (e) => {
            e.stopPropagation();
            startResizing(e, block);
        });
        
        // Add drag functionality to the block itself (excluding resize handle)
        block.addEventListener('mousedown', (e) => {
            // Only start dragging if we're not clicking on the resize handle
            if (!e.target.classList.contains('resize-handle')) {
                e.preventDefault(); // Prevent default to avoid text selection during drag
                e.stopPropagation();
                
                isDragging = true;
                selectedBlock = block;
                selectBlock(selectedBlock);
                
                dragStartX = e.clientX;
                dragStartY = e.clientY;
                // Calculate the original block position relative to the canvas
                originalX = getNumericStyleValue(block, 'left', 0);
                originalY = getNumericStyleValue(block, 'top', 0);
            }
        });
        
        // Add content editing for text blocks (initially disabled)
        if (blockData.type === 'text') {
            blockContent.addEventListener('blur', () => {
                disableBlockEditing(block);
            });
            
            blockContent.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    disableBlockEditing(block);
                }
            });
        }
    }
    
    // Handle canvas click
    function handleCanvasClick(e) {
        // Deselect block when clicking on canvas
        if (selectedBlock) {
            deselectBlock();
        }
    }
    
    // Select and edit a block
    function selectAndEditBlock(block) {
        // Select the block
        selectBlock(block);
        
        // For text blocks, enable editing
        if (block.classList.contains('text')) {
            enableBlockEditing(block);
        }
    }
    
    // Select a block with selection highlight
    function selectBlock(block) {
        // Deselect previous block
        if (selectedBlock) {
            deselectBlock();
        }
        
        // Select new block
        selectedBlock = block;
        block.classList.add('selected');
        
        // Add selection animation
        block.style.transform = 'translateY(-2px)';
        block.style.boxShadow = '0 0 0 3px rgba(67, 97, 238, 0.3), 0 10px 30px rgba(0, 0, 0, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.4)';
        
        // Slightly lighten the background when selected
        const currentBg = getComputedStyle(block).backgroundColor;
        if (currentBg.includes('rgba')) {
            block.style.background = currentBg.replace(/[\d.]+\)/, '0.2)');
        }
        
        // Show properties panel
        showBlockProperties(block);
    }
    
    // Deselect current block
    function deselectBlock() {
        if (selectedBlock) {
            selectedBlock.classList.remove('selected');
            selectedBlock.style.transform = 'translateY(0)';
            selectedBlock.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.2)';
            
            // Restore original background
            const blockData = {
                backgroundColor: getComputedStyle(selectedBlock).backgroundColor
            };
            selectedBlock.style.background = blockData.backgroundColor;
                
            selectedBlock = null;
        }
        
        // Show canvas properties panel
        showCanvasProperties();
    }
    
    // Enable editing for a block
    function enableBlockEditing(block) {
        if (!block) return;
        
        const blockContent = block.querySelector('.block-content');
        if (!blockContent) return;
        
        // Make content editable
        blockContent.contentEditable = true;
        blockContent.focus();
        
        // Select all text for easy replacement
        const range = document.createRange();
        range.selectNodeContents(blockContent);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        
        // Add editing class for visual feedback
        block.classList.add('editing');
        
        // Add event listener to detect when editing is finished
        blockContent.addEventListener('blur', finishEditing);
        blockContent.addEventListener('keydown', handleEditingKeys);
    }
    
    // Disable editing for a block
    function disableBlockEditing(block) {
        if (!block) return;
        
        const blockContent = block.querySelector('.block-content');
        if (!blockContent) return;
        
        // Make content non-editable
        blockContent.contentEditable = false;
        
        // Remove editing class
        block.classList.remove('editing');
        
        // Remove event listeners
        blockContent.removeEventListener('blur', finishEditing);
        blockContent.removeEventListener('keydown', handleEditingKeys);
        
        // Update block content on server
        updateBlockContent(block.id, blockContent.innerHTML);
    }
    
    // Finish editing when focus is lost
    function finishEditing(e) {
        const block = e.target.closest('.block');
        if (block) {
            disableBlockEditing(block);
        }
    }
    
    // Handle key events during editing
    function handleEditingKeys(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            const block = e.target.closest('.block');
            if (block) {
                disableBlockEditing(block);
            }
        } else if (e.key === 'Escape') {
            const block = e.target.closest('.block');
            if (block) {
                // Revert to original content
                const blockContent = block.querySelector('.block-content');
                // We would need to store original content to revert properly
                disableBlockEditing(block);
            }
        }
    }
    
    // Show block properties in the panel
    function showBlockProperties(block) {
        blockPropertiesPanel.classList.remove('hidden');
        canvasPropertiesPanel.classList.add('hidden');
        
        // Populate properties
        const blockContent = block.querySelector('.block-content');
        
        const leftValue = Math.round(getNumericStyleValue(block, 'left', 0));
        const topValue = Math.round(getNumericStyleValue(block, 'top', 0));
        const widthValue = Math.round(Math.max(MIN_BLOCK_WIDTH, getNumericStyleValue(block, 'width', MIN_BLOCK_WIDTH)));
        const heightValue = Math.round(Math.max(MIN_BLOCK_HEIGHT, getNumericStyleValue(block, 'height', MIN_BLOCK_HEIGHT)));
        
        blockContentInput.value = blockContent.innerHTML;
        blockXInput.value = leftValue;
        blockYInput.value = topValue;
        blockWidthInput.value = widthValue;
        blockHeightInput.value = heightValue;
        blockBgColorInput.value = rgbToHex(getComputedStyle(block).backgroundColor) || '#ffffff';
    }
    
    // Show canvas properties in the panel
    function showCanvasProperties() {
        blockPropertiesPanel.classList.add('hidden');
        canvasPropertiesPanel.classList.remove('hidden');
    }
    
    // Update block properties from panel inputs
    function updateBlockProperties() {
        if (!selectedBlock) return;
        
        const blockContent = selectedBlock.querySelector('.block-content');
        
        // Update content
        blockContent.innerHTML = blockContentInput.value;
        
        // Parse input values, defaulting to 0 if NaN
        const left = toNumber(blockXInput.value, Number.NaN);
        const top = toNumber(blockYInput.value, Number.NaN);
        const width = toNumber(blockWidthInput.value, Number.NaN);
        const height = toNumber(blockHeightInput.value, Number.NaN);
        
        if (!Number.isFinite(left) || !Number.isFinite(top) || !Number.isFinite(width) || !Number.isFinite(height)) {
            return;
        }
        
        const clampedWidth = Math.max(MIN_BLOCK_WIDTH, width);
        const clampedHeight = Math.max(MIN_BLOCK_HEIGHT, height);
        
        // Update position and size
        const roundedLeft = Math.round(left);
        const roundedTop = Math.round(top);
        const roundedWidth = Math.round(clampedWidth);
        const roundedHeight = Math.round(clampedHeight);
        
        selectedBlock.style.left = roundedLeft + 'px';
        selectedBlock.style.top = roundedTop + 'px';
        selectedBlock.style.width = roundedWidth + 'px';
        selectedBlock.style.height = roundedHeight + 'px';
        blockXInput.value = roundedLeft;
        blockYInput.value = roundedTop;
        blockWidthInput.value = roundedWidth;
        blockHeightInput.value = roundedHeight;
        
        // Update block on server
        updateBlockOnServer(selectedBlock.id, {
            content: blockContent.innerHTML,
            position: {
                left: roundedLeft,
                top: roundedTop,
                width: roundedWidth,
                height: roundedHeight
            }
        });
    }
    
    // Delete selected block
    function deleteSelectedBlock() {
        if (!selectedBlock) return;
        
        const blockId = selectedBlock.id;
        
        // Add deletion animation
        selectedBlock.style.transform = 'scale(0)';
        selectedBlock.style.opacity = '0';
        
        // Remove from DOM after animation
        setTimeout(() => {
            if (selectedBlock) {
                selectedBlock.remove();
                selectedBlock = null;
                
                // Show canvas properties
                showCanvasProperties();
                
                // Delete from server
                deleteBlockOnServer(blockId);
            }
        }, 300);
    }
    
    // Handle mouse move
    function handleMouseMove(e) {
        if (isDragging && selectedBlock) {
            // Calculate mouse movement
            const dx = e.clientX - dragStartX;
            const dy = e.clientY - dragStartY;
            
            // Calculate new position (accounting for zoom level)
            const newLeft = originalX + (dx / zoomLevel);
            const newTop = originalY + (dy / zoomLevel);
            
            selectedBlock.style.left = newLeft + 'px';
            selectedBlock.style.top = newTop + 'px';
            
            // Update property panel with proper integer values
            if (selectedBlock === selectedBlock) {
                blockXInput.value = Math.round(newLeft);
                blockYInput.value = Math.round(newTop);
            }
        }
        
        if (isResizing && selectedBlock) {
            // Calculate mouse movement for resizing
            const dx = e.clientX - dragStartX;
            const dy = e.clientY - dragStartY;
            
            // Calculate new dimensions (accounting for zoom level)
            const newWidth = Math.max(MIN_BLOCK_WIDTH, originalWidth + (dx / zoomLevel));  // Enforce minimum width
            const newHeight = Math.max(MIN_BLOCK_HEIGHT, originalHeight + (dy / zoomLevel)); // Enforce minimum height
            
            selectedBlock.style.width = newWidth + 'px';
            selectedBlock.style.height = newHeight + 'px';
            
            // Update property panel with proper integer values
            if (selectedBlock === selectedBlock) {
                blockWidthInput.value = Math.round(newWidth);
                blockHeightInput.value = Math.round(newHeight);
            }
        }
    }
    
    // Handle mouse up
    function handleMouseUp() {
        if (isDragging && selectedBlock) {
            // Update block position on server
            const left = Math.round(getNumericStyleValue(selectedBlock, 'left', 0));
            const top = Math.round(getNumericStyleValue(selectedBlock, 'top', 0));
            updateBlockPosition(selectedBlock.id, { left, top });
            isDragging = false;
        }
        
        if (isResizing && selectedBlock) {
            // Update block size on server
            const width = Math.round(Math.max(MIN_BLOCK_WIDTH, getNumericStyleValue(selectedBlock, 'width', MIN_BLOCK_WIDTH)));
            const height = Math.round(Math.max(MIN_BLOCK_HEIGHT, getNumericStyleValue(selectedBlock, 'height', MIN_BLOCK_HEIGHT)));
            updateBlockSize(selectedBlock.id, { width, height });
            isResizing = false;
        }
    }
    
    // Start resizing a block
    function startResizing(e, block) {
        e.stopPropagation();
        
        isResizing = true;
        selectedBlock = block;
        selectBlock(selectedBlock);
        
        dragStartX = e.clientX;
        dragStartY = e.clientY;
        originalWidth = getNumericStyleValue(block, 'width', MIN_BLOCK_WIDTH);
        originalHeight = getNumericStyleValue(block, 'height', MIN_BLOCK_HEIGHT);
    }
    
    // Update block position on server
    async function updateBlockPosition(blockId, position) {
        try {
            const response = await fetch('/api/block', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    operation: 'update',
                    block_id: blockId,
                    updates: {
                        position: {
                            ...getBlockPosition(blockId),
                            ...position
                        }
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (!data.success) {
                console.error('Failed to update block position:', data.error);
            }
        } catch (error) {
            console.error('Error updating block position:', error);
        }
    }
    
    // Update block size on server
    async function updateBlockSize(blockId, size) {
        try {
            const response = await fetch('/api/block', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    operation: 'update',
                    block_id: blockId,
                    updates: {
                        position: {
                            ...getBlockPosition(blockId),
                            ...size
                        }
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (!data.success) {
                console.error('Failed to update block size:', data.error);
            }
        } catch (error) {
            console.error('Error updating block size:', error);
        }
    }
    
    // Update block content on server
    async function updateBlockContent(blockId, content) {
        try {
            const response = await fetch('/api/block', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    operation: 'update',
                    block_id: blockId,
                    updates: {
                        content: content
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (!data.success) {
                console.error('Failed to update block content:', data.error);
            }
        } catch (error) {
            console.error('Error updating block content:', error);
        }
    }
    
    // Update block on server with all properties
    async function updateBlockOnServer(blockId, updates) {
        try {
            const response = await fetch('/api/block', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    operation: 'update',
                    block_id: blockId,
                    updates: updates
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (!data.success) {
                console.error('Failed to update block:', data.error);
            }
        } catch (error) {
            console.error('Error updating block:', error);
        }
    }
    
    // Delete block on server
    async function deleteBlockOnServer(blockId) {
        try {
            const response = await fetch('/api/block', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    operation: 'delete',
                    block_id: blockId
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (!data.success) {
                console.error('Failed to delete block:', data.error);
            }
        } catch (error) {
            console.error('Error deleting block:', error);
        }
    }
    
    // Get block position
    function getBlockPosition(blockId) {
        const block = document.getElementById(blockId);
        if (block) {
            return {
                left: Math.round(getNumericStyleValue(block, 'left', 0)),
                top: Math.round(getNumericStyleValue(block, 'top', 0)),
                width: Math.round(Math.max(MIN_BLOCK_WIDTH, getNumericStyleValue(block, 'width', MIN_BLOCK_WIDTH))),
                height: Math.round(Math.max(MIN_BLOCK_HEIGHT, getNumericStyleValue(block, 'height', MIN_BLOCK_HEIGHT)))
            };
        }
        return { left: 0, top: 0, width: 0, height: 0 };
    }
    
    // Zoom in
    function zoomIn() {
        zoomLevel = Math.min(zoomLevel + zoomStep, 2);
        applyZoom();
    }
    
    // Zoom out
    function zoomOut() {
        zoomLevel = Math.max(zoomLevel - zoomStep, 0.5);
        applyZoom();
    }
    
    // Apply zoom to canvas
    function applyZoom() {
        canvas.style.transform = `scale(${zoomLevel})`;
        canvas.style.transformOrigin = 'top left';
        updateZoomDisplay();
    }
    
    // Update zoom level display
    function updateZoomDisplay() {
        zoomLevelDisplay.textContent = `${Math.round(zoomLevel * 100)}%`;
    }
    
    // Clear canvas
    function clearCanvas() {
        if (confirm('Are you sure you want to clear the canvas? This cannot be undone.')) {
            // Add animation to all blocks before removing
            const blocks = canvas.querySelectorAll('.block');
            blocks.forEach(block => {
                block.style.transform = 'scale(0)';
                block.style.opacity = '0';
            });
            
            // Clear after animation
            setTimeout(() => {
                canvas.innerHTML = '<div class="canvas-grid"></div>';
                showCanvasProperties();
            }, 300);
        }
    }
    
    // Save layout to server
    async function saveLayout() {
        try {
            const layout = serializeLayout();
            const response = await fetch('/api/layout', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ layout })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            showNotification('Layout saved successfully!', 'success');
        } catch (error) {
            console.error('Error saving layout:', error);
            showNotification('Failed to save layout', 'error');
        }
    }
    
    // Load layout from server
    async function loadLayout() {
        try {
            const response = await fetch('/api/layout');
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            applyLayout(data);
            showCanvasProperties();
            showNotification('Layout loaded successfully!', 'success');
        } catch (error) {
            console.error('Error loading layout:', error);
            // Use default layout
            applyLayout(getDefaultLayout());
            showNotification('Loaded default layout', 'info');
        }
    }
    
    // Serialize the current layout
    function serializeLayout() {
        const blocks = [];
        const blockElements = canvas.querySelectorAll('.block');
        
        blockElements.forEach(block => {
            const blockContent = block.querySelector('.block-content');
            blocks.push({
                id: block.id,
                type: block.classList.contains('text') ? 'text' : 'image',
                content: blockContent.innerHTML,
                position: {
                    left: Math.round(getNumericStyleValue(block, 'left', 0)),
                    top: Math.round(getNumericStyleValue(block, 'top', 0)),
                    width: Math.round(Math.max(MIN_BLOCK_WIDTH, getNumericStyleValue(block, 'width', MIN_BLOCK_WIDTH))),
                    height: Math.round(Math.max(MIN_BLOCK_HEIGHT, getNumericStyleValue(block, 'height', MIN_BLOCK_HEIGHT)))
                }
            });
        });
        
        return {
            blocks: blocks,
            columns: 3,
            baseline: 24,
            gutter: 32,
            snap: true,
            zoom: zoomLevel,
            orientation: 'portrait',
            format: 'A4',
            dimensions: { width: 794, height: 1123 },
            layers: [
                {
                    id: 'layer-main',
                    name: 'Layer 1',
                    order: 0,
                }
            ],
            activeLayer: 'layer-main'
        };
    }
    
    // Apply layout to canvas
    function applyLayout(layout) {
        // Clear existing blocks (but keep the grid)
        const grid = canvas.querySelector('.canvas-grid');
        canvas.innerHTML = '';
        if (grid) {
            canvas.appendChild(grid);
        }
        
        // Set zoom level
        if (layout.zoom) {
            zoomLevel = layout.zoom;
            applyZoom();
        }
        
        // Add blocks from layout
        if (layout.blocks) {
            layout.blocks.forEach(blockData => {
                renderBlock(blockData);
            });
        }
    }
    
    // Get default layout
    function getDefaultLayout() {
        return {
            columns: 3,
            baseline: 24,
            gutter: 32,
            snap: true,
            zoom: 1.0,
            orientation: 'portrait',
            format: 'A4',
            dimensions: { width: 794, height: 1123 },
            blocks: [],
            layers: [
                {
                    id: 'layer-main',
                    name: 'Layer 1',
                    order: 0,
                }
            ],
            activeLayer: 'layer-main'
        };
    }
    
    // Helper function to convert RGB to Hex
    function rgbToHex(rgb) {
        if (!rgb) return '#ffffff';
        
        // Check if it's already a hex value
        if (rgb.startsWith('#')) return rgb;
        
        // Convert rgb() to hex
        const match = rgb.match(/^rgb\((\d+),\s*(\d+),\s*(\d+)\)$/);
        if (!match) return '#ffffff';
        
        const r = parseInt(match[1]).toString(16).padStart(2, '0');
        const g = parseInt(match[2]).toString(16).padStart(2, '0');
        const b = parseInt(match[3]).toString(16).padStart(2, '0');
        
        return `#${r}${g}${b}`;
    }
    
    // Helper function to get random position
    function getRandomPosition(min, max) {
        return Math.floor(Math.random() * (max - min + 1)) + min;
    }
    
    // Helper function to get random size
    function getRandomSize(min, max) {
        return Math.floor(Math.random() * (max - min + 1)) + min;
    }
    
    // Show notification
    function showNotification(message, type) {
        // Remove existing notifications
        const existing = document.querySelector('.notification');
        if (existing) {
            existing.remove();
        }
        
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 20px;
            border-radius: 50px;
            color: white;
            font-size: 14px;
            font-weight: 500;
            z-index: 10000;
            opacity: 0;
            transform: translateX(100%);
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        `;
        
        // Set background color based on type
        if (type === 'success') {
            notification.style.background = 'linear-gradient(135deg, #06d6a0, #05b38a)';
        } else if (type === 'error') {
            notification.style.background = 'linear-gradient(135deg, #ef476f, #d92655)';
        } else {
            notification.style.background = 'linear-gradient(135deg, #4361ee, #3a56d4)';
        }
        
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.style.opacity = '1';
            notification.style.transform = 'translateX(0)';
        }, 10);
        
        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.opacity = '0';
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }
    
    // Add CSS for animations
    const style = document.createElement('style');
    style.textContent = `
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .notification {
            animation: fadeIn 0.3s ease-out;
        }
    `;
    document.head.appendChild(style);
    
    // Initialize the application
    initApp();
    
    // Pages functionality
    // This will be implemented in a separate file
});
