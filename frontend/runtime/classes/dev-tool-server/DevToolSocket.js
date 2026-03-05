'use strict';

import { DevToolSocketLog } from './DevToolSocketLog.js';
import { Log } from '../Log.js';

export class DevToolSocket {
    
    constructor(devToolServer, socket, request) {

        this.devToolServer = devToolServer;
        this.socket = socket;
        this.request = request;

        this.runtime = this.devToolServer.runtime;

        this.logger = new DevToolSocketLog(this);
        this.logger.log('Connected')

        this.socket.on('message', this.onMessage.bind(this));
        this.socket.on('close', this.onClose.bind(this));

        this.sendTypeCheckingStatus();
        this.sendBundlingSystem();
        this.sendRuntimeStatus();
    }

    /*
    **
    **
    */
    onMessage(json) {

        let data;

        try {
            
            data = JSON.parse(json);

            if (typeof data !== 'object'
             || data === null
             || typeof data.type !== 'string')
                throw new Error('Bad message format');   
        }
        catch(error) {
            Log.red('DevToolSocket onMesage error');
            Log.printError(error);
            return;
        }

        switch (data.type) {

            case 'toggle-type-checking': 
            this.onToggleTypeCheckingRequest(data);
            break;

            case 'toggle-bundling-system':
            this.onToggleBundlingSystem(data);
            break;

            case 'build':
            this.onBuildRequest(data);
            break;
        }
    }

    /*
    **
    **
    */
    onToggleTypeCheckingRequest() {

        this.runtime.toggleTypeChecking();

        this.logger.log(`Type checking ${this.runtime.typeCheckingStatus === 'on' ? 'enabled' : 'disabled'}`);
    }

    /*
    **
    **
    */
    onToggleBundlingSystem() {

        this.runtime.toggleBundlingSystem();

        this.logger.log(`Bundling system set to ${this.runtime.bundlingSystem}`);
    }

    /*
    **
    **
    */
    onBuildRequest() {

        this.runtime.devRefresh();

        this.logger.log(`Build requested`);
    }

    /*
    **
    **
    */
    onClose() {

        this.logger.log('Disconnected');
    }

    /*
    **
    **
    */
    getRemoteIp() {

        if (this.request.headers['x-real-ip'])
            return this.request.headers['x-real-ip'];
        
        let ip = this.request.socket.remoteAddress;
        
        if (ip) {
            if (ip.slice(0, 7) === '::ffff:')
                ip = ip.slice(7);
        }
        else 
            ip = 'undefined';
            
        return ip ? ip : '';
    }

    /*
    **
    **
    */
    getRemoteFamily() {

        if (/^((25[0-5]|(2[0-4]|1[0-9]|[1-9]|)[0-9])(\.(?!$)|$)){4}$/.test(this.getRemoteIp()))
            return 'IPv4';
        else
            return 'IPv6';
    }

    /*
    **
    **
    */
    send(type, data={}) {

        const json = JSON.stringify({
            type: type,
            data: data
        });

        this.socket.send(json);
    }

    /*
    **
    **
    */
    sendTypeCheckingStatus() {

        this.send('type-checking-status', {
            status: this.runtime.typeChecking ? 'on' : 'off'
        });
    }

    /*
    **
    **
    */
    sendBundlingSystem() {

        this.send('bundling-system', {
            system: this.runtime.bundlingSystem
        });
    }
    
    /*
    **
    **
    */
    sendRuntimeStatus() {

        this.send('runtime-status', {
            status: this.runtime.status
        });
    }
}