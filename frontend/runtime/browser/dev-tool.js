'use strict';

class DevTool {

    constructor() {

        this.element = document.createElement('div');
        this.element.setAttribute('id', 'dev-tool');
        this.element.setAttribute('data-socket-status', 'pending');
        this.element.setAttribute('data-type-checking-status', 'pending');
        this.element.setAttribute('data-runtime-status', 'pending');

        this.ui = document.createElement('div');
        this.ui.classList.add('ui');
        
        this.socketStatus = document.createElement('div');
        this.socketStatus.classList.add('socket-status');
        
        this.typeCheckingStatus = document.createElement('div');
        this.typeCheckingStatus.classList.add('type-checking-status');

        this.bundlingSystem = document.createElement('div');
        this.bundlingSystem.classList.add('bundling-system');

        this.runtimeStatus = document.createElement('div');
        this.runtimeStatus.classList.add('runtime-status');
        
        this.ui.append(this.socketStatus);
        this.ui.append(this.typeCheckingStatus);
        this.ui.append(this.bundlingSystem);
        this.ui.append(this.runtimeStatus);
        this.element.append(this.ui);
        document.body.append(this.element);
        
        this.client = new DevToolClient(this);

        this.socketStatus.addEventListener('click', this.onSocketStatusClick.bind(this));
        this.typeCheckingStatus.addEventListener('click', this.onTypeCheckingStatusClick.bind(this));
        this.bundlingSystem.addEventListener('click', this.onBundlingSystemClick.bind(this));
        this.runtimeStatus.addEventListener('click', this.onRuntimeStatusClick.bind(this));
    }

    /*
    **
    **
    */
    setSocketStatus(status) {

        this.element.setAttribute('data-socket-status', status);
    }

    /*
    **
    **
    */
    setTypeCheckingStatus(status) {

        this.element.setAttribute('data-type-checking-status', status);
    }

    /*
    **
    **
    */
    setBundlingSystem(system) {

        this.element.setAttribute('data-bundling-system', system);
    }

    /*
    **
    **
    */
    setRuntimeStatus(status) {

        this.element.setAttribute('data-runtime-status', status);
    }

    /*
    **
    **
    */
    onSocketStatusClick(event) {

        if (this.client.socket)
            this.client.closeAndLock();
        else {
            this.setSocketStatus('opening');
            this.client.unlock();
        }
    }

    /*
    **
    **
    */
    onTypeCheckingStatusClick(event) {

        if (!this.client.canSend())
            return;
            
        this.client.send('toggle-type-checking');
    }

    /*
    **
    **
    */
    onBundlingSystemClick(event) {

        if (!this.client.canSend())
            return;
            
        this.client.send('toggle-bundling-system');
    }

    /*
    **
    **
    */
    onRuntimeStatusClick(event) {

        if (!this.client.canSend())
            return;

        this.setRuntimeStatus('building');
        
        this.client.send('build');
    }

    /*
    **
    **
    */
    isBuilding() {

        return this.element.getAttribute('data-runtime-status') === 'building';
    }
}

class DevToolClient {

    constructor(devTool) {

        this.devTool = devTool;
        this.lock = false;

        this.autoOpen();
    }

    /*
    **
    **
    */
    autoOpen() {

        this.open();

        setTimeout(this.autoOpen.bind(this), 1000);
    }

    /*
    **
    **
    */
    open() {

        if (this.opening || this.lock || (this.socket && this.socket.readyState !== 3))
            return;

        this.opening = true;

        this.socket = new WebSocket(`ws${document.location.protocol === "https:" ? 's' : ''}://${document.location.hostname}/dev-tool`);
        
        this.socket.addEventListener('open', this.onOpen.bind(this));
        this.socket.addEventListener('close', this.onClose.bind(this));
        this.socket.addEventListener('message', this.onMessage.bind(this));
        this.socket.addEventListener('error', function() {
            this.socket = null;
            this.opening = false;
        }.bind(this));
    }

    /*
    **
    **
    */
    closeAndLock() {

        this.socket.close();
        this.lock = true;
        this.socket = null;
    }

    /*
    **
    **
    */
    unlock() {

        this.lock = false;
    }

    /*
    **
    **
    */
    send(type, data={}) {

        this.socket.send(JSON.stringify({
            type: type,
            data: data
        }));
    }
    
    /*
    **
    **
    */
    canSend() {

        return this.socket && this.socket.readyState === 1;
    }

    /*
    **
    **
    */
    onOpen(event) {

        this.opening = false;

        this.devTool.setSocketStatus('opened');
    }

    /*
    **
    **
    */
    onClose(event) {

        this.devTool.setSocketStatus(this.lock ? 'pending' : 'closed');
        this.devTool.setTypeCheckingStatus('off');
        this.devTool.setRuntimeStatus('pending');
    }

    /*
    **
    **
    */
    onMessage(event) {

        let message;

        try {
            message = JSON.parse(event.data)
        }
        catch(error) {
            console.error(error);
            return;
        }

        switch (message.type) {

            case 'ping':
                this.onPing(message.data);
                break;

            case 'type-checking-status':
                this.onTypeCheckingStatus(message.data);
                break;
            
            case 'bundling-system':
                this.onBundlingSystem(message.data);
                break;

            case 'runtime-status':
                this.onRuntimeStatus(message.data);
                break;

            case 'reload':
                this.onReloadRequest(message.data);
                break;
        }
    }

    /*
    **
    **
    */
    onPing(data) {

        this.send('pong');
    }

    /*
    **
    **
    */
    onTypeCheckingStatus(data) {

        if (['on', 'off'].includes(data.status))
            this.devTool.setTypeCheckingStatus(data.status);
    }

    /*
    **
    **
    */
    onBundlingSystem(data) {

        if (['fast', 'sourcemap', 'production'].includes(data.system))
            this.devTool.setBundlingSystem(data.system);
    }

    /*
    **
    **
    */
    onRuntimeStatus(data) {

        if (['done', 'building', 'error'].includes(data.status))
            this.devTool.setRuntimeStatus(data.status);
    }

    /*
    **
    **
    */
    onReloadRequest(data) {

        location.reload();
    }
}

window.addEventListener('DOMContentLoaded', function() {

    new DevTool();
});