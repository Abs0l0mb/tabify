'use strict';

import {
    Emitter,
    HttpRequest,
    WebSocketInputRequest,
    WebSocketInputMessage,
    Tools,
    Log
} from '@src/classes';

import { WebSocket } from 'ws';

export class WebSocketClient extends Emitter {

    private key: string = Tools.uid(32);

    constructor(private socket: WebSocket, private request: HttpRequest) {

        super();

        socket.on('message', this.onInput.bind(this));
        socket.on('close', this.onClose.bind(this));
    }

    /*
    **
    **
    */
    public getSocket() : WebSocket {

        return this.socket;
    }

    /*
    **
    **
    */
    public getRequest() : HttpRequest {

        return this.request;
    }

    /*
    **
    **
    */
    public getKey() : string {

        return this.key;
    }

    /*
    **
    **
    */
    private onInput(buffer: Buffer) : void {

        this.emit('input', buffer);
    }

    /*
    **
    **
    */
    private onClose() : void {

        this.emit('close');
    }

    /*
    **
    **
    */
    public log(message: string) : void {

        const remoteFamily = this.getRequest().getRemoteFamily();
        const remoteAddress = this.getRequest().getRemoteIp();
        const remotePort = this.getRequest().getRemotePort();
        const builtUrl = this.getRequest().getURL();

        let remoteSocket = '';
        let url = '';
        let channel = '';
        let account = '';
        let date = '';

        if (remoteFamily === 'IPv6') {
            if (remoteAddress && remoteAddress.indexOf('::ffff:') === 0)
                remoteSocket = `${Log.Dim}[::ffff:${Log.Reset}${Log.Bright}${remoteAddress.slice(7)}${Log.Reset}${Log.Dim}]:${remotePort}${Log.Reset}`;
            else
                remoteSocket = `${Log.Dim}[${Log.Reset}${Log.Bright}${remoteAddress}${Log.Reset}${Log.Dim}]:${remotePort}${Log.Reset}`;
        }
        else
            remoteSocket = `${Log.FgGreen}${Log.Bright}${remoteAddress}${Log.Reset}${Log.FgGreen}${Log.Dim}:${remotePort}${Log.Reset}`;

        let software: string | null = null;
        let pua = this.getRequest().getParsedUserAgent();
        let os = pua.os ? pua.os.name : '';
        let browser = pua.browser ? `${pua.browser.name} ${pua.browser.major}` : '';

        if (os && browser)
            software = `${Log.Dim}${os} ${browser}${Log.Reset}`;
        
        account = this.getRequest().account ? `[${this.getRequest().account.id}]` : '';
        const method = `${Log.FgCyan}${Log.Dim}UPGRADED${Log.Reset}`;
        url = `${Log.FgCyan}${Log.Dim}${builtUrl.pathname}${Log.Reset}${Log.FgCyan}${Log.Dim}${builtUrl.search}${Log.Reset}`;
        message = `${Log.FgCyan}${Log.Bright}${message}${Log.Reset}`;
        date = `${Log.Dim}${new Date().toLocaleString('fr')}${Log.Reset}`;

        Log.log(`${remoteSocket} ${software} ${account} ${method} ${url} ${message}`.trim());
    }

    /*
    **
    **
    */
    public logRequest(request: WebSocketInputRequest) : void {

        const parameters =  `${Log.FgYellow}${Log.Dim}${JSON.stringify(this.getLightParameters(request.getData()))}${Log.Reset}`;
        const info = `${Log.Dim}>${Log.Reset} ${request.getError() ? Log.FgRed : Log.FgGreen}${Log.Bright}${request.getError() ? request.getError() : 'ok'}${Log.Reset}`;
        const duration = `${Log.Dim}[${request.getDuration()} ms]${Log.Reset}`;
        
        this.log(`${Log.FgYellow}${Log.Bright}REQUEST ${request.getTopic()}${Log.Reset} ${parameters} ${info} ${duration}`.trim());
    }

    /*
    **
    **
    */
    public logMessage(message: WebSocketInputMessage) : void {

        const parameters =  `${Log.FgYellow}${Log.Dim}${JSON.stringify(this.getLightParameters(message.getData()))}${Log.Reset}`;

        this.log(`${Log.FgYellow}${Log.Bright}MESSAGE ${message.getTopic()}${Log.Reset} ${parameters}`.trim());
    }

    /*
    **
    **
    */
    public logError(errorMessage: string) : void {

        this.log(`${Log.FgRed}${errorMessage}${Log.Reset}`);
    }

    /*
    **
    **
    */
    private getLightParameters(parameters: any) : any {

        let output: any = {};

        for (let key in parameters) {
            if (key.toLowerCase() === 'password')
                output[key] = '***';
            else if (typeof parameters[key] === 'object' && parameters[key] !== null) 
                output[key] = this.getLightParameters(parameters[key]);
            else if (parameters[key] === null)
                output[key] = null;
            else {

                let isNumber = typeof parameters[key] === 'number';

                let test = parameters[key].toString();

                if (test.length > 256)
                    test = test.slice(0, 256) + '...';
                
                if (isNumber)
                    test = new Number(test).valueOf();

                output[key] = test;
            }
        }

        return output;
    }
}