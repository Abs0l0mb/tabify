'use strict';

import {
    IPCRequest,
    IPCResponseCallback,
    IPCRawRequest,
    IPCRawResponse,
    Log
} from '@src/classes';

import * as net from 'net';

export class IPCServerSocket {

    private buffer: Buffer = Buffer.alloc(0);

    constructor(private socket: net.Socket, private port: number, private onRequest: (request: IPCRequest) => void) {

        socket.on('data', this.onData.bind(this));
        socket.on('error', this.onError.bind(this));
        socket.on('close', this.onClose.bind(this)); 
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

                const rawRequest: IPCRawRequest = JSON.parse(messageBuffer.toString('utf-8'));

                if (!rawRequest.hasOwnProperty('topic')
                 || !(typeof rawRequest.topic === 'string')
                 || !rawRequest.hasOwnProperty('data'))
                    return;
       
                let responseCallback: IPCResponseCallback | null = null;
                
                if (rawRequest.hasOwnProperty('id') && typeof rawRequest.id === 'string') {
    
                    responseCallback = async (data: any) : Promise<void> => {
    
                        await this.respondTo(rawRequest.id!, data);
                    }
                }
    
                const request: IPCRequest = new IPCRequest(rawRequest.topic, rawRequest.data, responseCallback ? responseCallback : undefined);
    
                this.onRequest(request);

            } else
                break;
        }
    }
    
    /*
    **
    **
    */
    private respondTo(id: string, data: any = {}) : Promise<void> {

        const rawResponse: IPCRawResponse = {
            id: id,
            data: data
        };

        return new Promise<void>((resolve, reject) => {

            if (this.socket.readyState === 'open') {

                this.socket.write(this.getResponseBuffer(rawResponse), () => {
                    resolve();
                });
            }
            else
                reject(new Error(`IPC socket not opened (${this.port})`));
        });
    }

    /*
    **
    **
    */
    private getResponseBuffer(rawResponse: IPCRawResponse) : Buffer {

        const message = JSON.stringify(rawResponse);

        const messageBuffer = Buffer.from(message, 'utf-8');
        const lengthBuffer = Buffer.alloc(4);
        lengthBuffer.writeUInt32BE(messageBuffer.length, 0);
        
        return Buffer.concat([lengthBuffer, messageBuffer]);
    }

    /*
    **
    **
    */
    private onError(error: Error) : void {

        Log.red(`IPC server socket error (${this.port})`);
        Log.printError(error);
    }
    
    /*
    **
    **
    */
    private onClose() : void {

        //Log.magenta(`IPC server: client disconnected`);

        this.buffer = Buffer.alloc(0);
    }
}