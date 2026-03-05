'use strict';

import {
    IPCRequest,
    IPCServerSocket,
    Log
} from '@src/classes';

import * as net from 'net';

export interface IPCRawRequest {
    id?: string,
    topic: string,
    data: any
}

export interface IPCRawResponse {
    id: string,
    data: any
}

export type IPCRequestCallback = (request: IPCRequest) => void;
export type IPCResponseCallback = (data: any) => Promise<void>;

export class IPCServer {

    private server: net.Server;
    private requestCallbacks: Map<string, IPCRequestCallback> = new Map();

    constructor(private port: number) {

        this.server = net.createServer(this.onSocket.bind(this));

        this.server.on('error', this.onError.bind(this));
    }

    /*
    **
    **
    */
    public async listen() : Promise<void> {

        await new Promise<void>(resolve => {

            this.server.listen(this.port, '127.0.0.1', resolve);
        });

        Log.magenta(`IPC server listening on ${this.port}`);
    }

    /*
    **
    **
    */
    private onSocket(socket: net.Socket) : void {

        //Log.magenta(`IPC server: client connected`);

        new IPCServerSocket(socket, this.port, this.onRequest.bind(this));
    }

    /*
    **
    **
    */
    public on(topic: string, requestCallback: IPCRequestCallback) {

        this.requestCallbacks.set(topic, requestCallback);
    }

    /*
    **
    **
    */
    protected onRequest(request: IPCRequest) : void {
        
        for (const [topic, requestCallback] of this.requestCallbacks) {

            if (topic === request.getTopic()) {

                try {
                    requestCallback(request);
                }
                catch(error) {
                    Log.red(`IPC request callback error (${this.port})`);
                    Log.printError(error);
                }
            }
        }
    }

    /*
    **
    **
    */
    private onError(error: Error) : void {

        Log.red(`IPC server error (${this.port})`);
        Log.printError(error);
    }
}