'use strict';

import * as ws from '../../lib/ws.cjs';

import { DevToolSocket } from './DevToolSocket.js';
import { Tools } from '../Tools.js';
import { Log } from '../Log.js';

export class DevToolServer {
    
    constructor(runtime) {

        this.runtime = runtime;
        this.sockets = {};
        
        this.wss = new ws.default.WebSocketServer({
            path: '/dev-tool',
            port: 10255
        });

        this.wss.on('listening', this.onListening.bind(this));
        this.wss.on('connection', this.onSocket.bind(this));

        this.ping();
    }
    
    /*
    **
    **
    */
    onListening() {

        Log.cyan(`DevToolServer listening`);
    }

    /*
    **
    **
    */
    onSocket(socket, request) {

        socket.id = Tools.randStr(5);

        this.sockets[socket.id] = new DevToolSocket(this, socket, request);
        
        socket.on('close', function() {
            delete this.sockets[socket.id];
        }.bind(this));
    }

    /*
    **
    **
    */
    ping() {

        this.broadcast('ping');

        setTimeout(this.ping.bind(this), 15000);
    }

    /*
    **
    **
    */
    broadcast(type, data={}) {
        
        const json = JSON.stringify({
            type: type,
            data: data
        });

        for (let client of this.wss.clients) {
        
            if (client.readyState === 1)
                client.send(json);
        }
    }

    /*
    **
    **
    */
    broadcastTypeCheckingStatus() {

        this.broadcast('type-checking-status', {
            status: this.runtime.typeCheckingStatus
        });
    }

    /*
    **
    **
    */
    broadcastBundlingSystem() {

        this.broadcast('bundling-system', {
            system: this.runtime.bundlingSystem
        });
    }
    
    /*
    **
    **
    */
    broadcastBuilderStatus() {

        this.broadcast('runtime-status', {
            status: this.runtime.status,
        });
    }
}