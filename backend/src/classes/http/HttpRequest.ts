'use strict';

import { IncomingMessage } from 'http';
import { parse } from 'querystring';
import { UAParser } from 'ua-parser-js';

export class HttpRequest extends IncomingMessage {

    private builtURL: URL;
    private builtHeaders: {[header: string]: string} = {}
    private parameters: any = {};
    private time: number = Date.now();

    public account: any;
    public session: any;

    private wsHead: Buffer = Buffer.alloc(0);

    /*
    **
    **
    */
    public getRemoteIp() : string | null {

        const realIp = this.getHeader('x-real-ip');

        if (realIp)
            return realIp;

        return this.socket.remoteAddress ? this.socket.remoteAddress : null;
    }

    /*
    **
    **
    */
    public getRemoteFamily() : string | null {

        const remoteIp = this.getRemoteIp();

        if (!remoteIp)
            return null;
        else if (/^((25[0-5]|(2[0-4]|1[0-9]|[1-9]|)[0-9])(\.(?!$)|$)){4}$/.test(remoteIp))
            return 'IPv4';
        else
            return 'IPv6';
    }

    /*
    **
    **
    */
    public getRemotePort() : number | null {

        return this.socket.remotePort ? this.socket.remotePort : null;
    }

    /*
    **
    **
    */
    public buildHeaders() : void {

        for (let i=0; i<this.rawHeaders.length; i=i+2)
            this.builtHeaders[this.rawHeaders[i].toLowerCase()] = this.rawHeaders[i+1];
    }

    /*
    **
    **
    */
    public getParsedUserAgent() : any {

        const userAgent = this.getHeader('user-agent');

        return (new UAParser(userAgent ? userAgent : '')).getResult();
    }

    /*
    **
    **
    */
    public buildURL() : void {
        
        try {

            this.builtURL = new URL(this.url as string, `http://${this.getHeader('host')}`);
        }
        catch(error) {

            this.builtURL = new URL(this.url as string, 'undefined://-');
        }

        if (this.builtURL.pathname.slice(-1) === '/')
            this.builtURL.pathname = this.builtURL.pathname.slice(0, -1);
    }

    /*
    **
    **
    */
    public async extractGETParameters() : Promise<void> {

        const url = this.getURL();
        
        if (!url)
            return;
            
        url.searchParams.forEach((value: string, key: string) => {

            try {
                value = JSON.parse(value);
            }
            catch(ignored) {
            }

            this.parameters[key] = value;
        });
    }

    /*
    **
    **
    */
    public async extractPOSTParameters() : Promise<void> {
        
        await new Promise((resolve: any, reject: any) => {
            
            const chunks: Buffer[] = [];

            this.on('data', function(chunk: Buffer) {
                chunks.push(chunk);
            });

            this.on('end', () => {

                const body = Buffer.concat(chunks).toString();

                try {

                    const parameters = parse(body);

                    for (const key in parameters) {

                        let value: any = parameters[key];

                        try {
                            value = JSON.parse(value);
                        }
                        catch(ignored) {
                        }

                        this.parameters[key] = value;
                    }

                    resolve();
                }
                catch(error) {

                    resolve();
                }
            });

        });
    }

    /*
    **
    **
    */
    public getHeader(key: string) : string | null {

        const header = this.builtHeaders[key.toLowerCase()];

        return header ? header : null;
    }

    /*
    **
    **
    */
    public getCookies() : any {

        const output: any = {};
        const raw = this.headers.cookie;

        if (!raw)
            return output;

        const cookies = raw.split('; ');

        for (const cookie of cookies) {
            const split = cookie.split('=');
            output[split[0]] = split[1];
        }

        return output;
    }

    /*
    **
    **
    */
    public getCookie(key: string) : string {

        return this.getCookies()[key];
    }

    /*
    **
    **
    */
    public getURL() : URL {

        return this.builtURL;
    }

    /*
    **
    **
    */
    public getParameters() : any {

        return this.parameters;
    }

    /*
    **
    **
    */
    public getTime() : number {

        return this.time;
    }

    /*
    **
    **
    */
    public wsAuthPatch() : void {

        if (this.getHeader('upgrade') === 'websocket') {
            const headerAuthKey = this.getHeader('sec-websocket-protocol');
            this.builtHeaders['x-stat'] = headerAuthKey ? headerAuthKey : '';
        }
    }

    /*
    **
    **
    */
    public cacheWsHead(head: Buffer) : void {

        this.wsHead = head;
    }

    /*
    **
    **
    */
    public getWsHead() : Buffer {

        return this.wsHead;
    }
}