'use strict';

import { IncomingMessage } from 'http';

export class Request extends IncomingMessage {

    constructor(...args) {

        super(...args);

        this.time = Date.now();

        this.builtHeaders = {};
    }

    /*
    **
    **
    */
    getRemoteIp() {

        const realIp = this.getHeader('x-real-ip');

        if (realIp)
            return realIp;

        return this.socket.remoteAddress ? this.socket.remoteAddress : null;
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
    decomposeUrl() {
        
        try {

            this.builtUrl = new URL(this.url, `http://${this.getHeader('host')}`);
        }
        catch(error) {

            this.builtUrl = new URL(this.url, 'undefined://-');
        }

        if (this.builtUrl.pathname.slice(-1) === '/')
            this.builtUrl.pathname = this.builtUrl.pathname.slice(0, -1);
    }

    /*
    **
    **
    */
    buildHeaders() {

        for (let i=0; i<this.rawHeaders.length; i=i+2)
            this.builtHeaders[this.rawHeaders[i].toLowerCase()] = this.rawHeaders[i+1];
    }

    /*
    **
    **
    */
    getHeader(key) {

        const header = this.builtHeaders[key.toLowerCase()];

        return header ? header : null;
    }
}