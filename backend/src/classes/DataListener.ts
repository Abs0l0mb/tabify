'use strict';

import {
    Log,
    Emitter,
    Tools
} from '@src/classes';

import { WebSocket } from 'ws';

export class DataListener extends Emitter {
    
    private client: WebSocket;

    constructor(private dataServer: string) {

        super();

        this.open();
    }

    /*
    **
    **
    */
    private open() : void {

        this.client = new WebSocket(this.dataServer);

        this.client.on('open', this.onOpen.bind(this));
        this.client.on('message', this.onMessage.bind(this));
        this.client.on('error', this.onError.bind(this));
        this.client.on('close', this.onClose.bind(this));
    }

    /*
    **
    **
    */
    private onOpen() : void {

        Log.green(`DataListener connected to ${this.dataServer}`);
    }

    /*
    **
    **
    */
    private onMessage(JSONMessage: string) : void {

        try {
            this.emit('data', JSON.parse(JSONMessage.toString()));
        }
        catch(error) {

            Log.printError(error);
        }
    }

    /*
    **
    **
    */
    private onError(error: any) : void {

        Log.printError(error);
    }

    /*
    **
    **
    */
    private async onClose() : Promise<void> {

        Log.red(`DataListener [${this.dataServer}] closed`);
        
        await Tools.sleep(2000);
        this.open()
    }
}