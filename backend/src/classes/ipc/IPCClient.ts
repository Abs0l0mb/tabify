'use strict';

import {
    IPCRawRequest,
    IPCRawResponse,
    Tools,
    Log
} from '@src/classes';

import * as net from 'net';

export interface IPCClientOptions {
    connectionTimeout?: number,
    requestTimeout?: number | null,
    reconnectionDelay?: number
}

export class IPCClient {

    static readonly DEFAULT_OPTIONS: IPCClientOptions = {
        requestTimeout: null,
        reconnectionDelay: 1000
    }

    private options: IPCClientOptions;
    private socket: net.Socket;
    private buffer: Buffer = Buffer.alloc(0);

    private responseCallbacks: Map<string, {
        timeout?: ReturnType<typeof setTimeout>,
        responseCallback: (data: any) => void
    }> = new Map();

    private requestingNewConnection: boolean = false;
    private resolveConnectionAttempt: () => void = () => {};

    constructor(private port: number, options: IPCClientOptions = {}) {

        this.options = Object.assign({}, IPCClient.DEFAULT_OPTIONS, options);

        this.checkConnectionRequest();
    }

    /*
    **
    **
    */
    public connect() : void {

        this.requestingNewConnection = true;
    }

    /*
    **
    **
    */
    public async checkConnectionRequest() : Promise<void> {

        if (this.requestingNewConnection) {

            this.requestingNewConnection = false;

            if (this.socket) {
                this.socket.removeAllListeners();
                this.socket.destroy();
            }

            this.socket = new net.Socket();
            
            this.socket.on('data', this.onData.bind(this));
            this.socket.on('error', this.onError.bind(this));
            this.socket.on('close', this.onClose.bind(this));

            this.buffer = Buffer.alloc(0);

            await new Promise<void>(resolve => {

                this.resolveConnectionAttempt = () => {
                    
                    this.resolveConnectionAttempt = () => {};

                    resolve();
                }
                
                this.socket.connect(this.port, '127.0.0.1', () => {

                    Log.magenta(`IPC client connected to ${this.port}`);

                    this.resolveConnectionAttempt();
                });
            });        
        }

        setTimeout(this.checkConnectionRequest.bind(this), 200);
    }

    /*
    **
    **
    */
    private onData(data: Buffer) : void {

        this.buffer = Buffer.concat([this.buffer, data]);

        while (this.buffer.length >= 4) {

            const messageLength = this.buffer.readUInt32BE(0);

            if (this.buffer.length >= 4 + messageLength) {

                const messageBuffer = this.buffer.subarray(4, 4 + messageLength);
                
                this.buffer = this.buffer.subarray(4 + messageLength);

                const rawResponse: IPCRawResponse = JSON.parse(messageBuffer.toString('utf-8'));

                if (!rawResponse.hasOwnProperty('id') 
                 || !(typeof rawResponse.id === 'string')
                 || !rawResponse.hasOwnProperty('data'))
                    return;
       
                this.onResponse(rawResponse.id, rawResponse.data);

            } else
                break;
        }
    }
    
    /*
    **
    **
    */
    public request(topic: string, data: any = {}) : Promise<any> {

        return new Promise((resolve, reject) => {
            
            const id = Tools.uid();

            let timeout: ReturnType<typeof setTimeout> | null = null;

            if (this.options.requestTimeout !== null) {
                const timeout = setTimeout(() => {
                    reject(new Error(`IPC request timeout (${this.port})`));
                    this.responseCallbacks.delete(id);
                }, this.options.requestTimeout);
            }

            const responseCallback = (data: any) => {
                resolve(data);
            };

            this.responseCallbacks.set(id, {
                timeout: timeout !== null ? timeout : undefined,
                responseCallback: responseCallback
            });

            const rawRequest: IPCRawRequest = {
                id: id,
                topic: topic,
                data: data
            };

            if (this.socket && this.socket.readyState === 'open')
                this.socket.write(this.getRequestBuffer(rawRequest));
            else
                reject(new Error(`IPC request error due to socket not opened (${this.port})`));
        });
    }

    /*
    **
    **
    */
    public send(topic: string, data: any = {}) : Promise<any> {

        return new Promise((resolve, reject) => {
            
            const rawRequest: IPCRawRequest = {
                topic: topic,
                data: data
            };

            if (this.socket && this.socket.readyState === 'open')
                this.socket.write(this.getRequestBuffer(rawRequest), resolve);
            else
                reject(new Error(`IPC send error due to socket not opened (${this.port})`));
        });
    }
    
    /*
    **
    **
    */
    private getRequestBuffer(rawRequest: IPCRawRequest) : Buffer {

        const message = JSON.stringify(rawRequest);

        const messageBuffer = Buffer.from(message, 'utf-8');
        const lengthBuffer = Buffer.alloc(4);
        lengthBuffer.writeUInt32BE(messageBuffer.length, 0);
        
        return Buffer.concat([lengthBuffer, messageBuffer]);
    }

    /*
    **
    **
    */
    private onResponse(responseId: string, data: any) : void {

        for (const [requestId, responseCallbackData] of this.responseCallbacks) {

            if (responseId === requestId) {

                try {
                    clearTimeout(responseCallbackData.timeout);
                    responseCallbackData.responseCallback(data);
                }
                catch(error) {
                    Log.red(`IPC response callback error (${this.port})`);
                    Log.printError(error);
                }
            }
        }
    }

    /*
    **
    **
    */
    private async onError(error: Error) : Promise<void> {

        Log.printError(error);
    }
    
    /*
    **
    **
    */
    private async onClose() : Promise<void> {

        this.resolveConnectionAttempt();

        Log.red(`IPC client socket closed ${this.port}`);

        await Tools.sleep(this.options.reconnectionDelay!);

        this.requestingNewConnection = true;
    }
}