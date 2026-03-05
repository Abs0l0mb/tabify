'use strict';

import { Log } from '../Log.js';
import { UAParser } from 'ua-parser-js';

export class DevToolSocketLog {

    constructor(devToolSocket) {

        this.devToolSocket = devToolSocket;
        this.request = this.devToolSocket.request;
    }

    /*
    **
    **
    */
    log(message) {

        Log.log(`${this.getRequestLog()} ${this.devToolSocket.socket.id} ${Log.FgCyan}${Log.Bright}${message}${Log.Reset}`);
    }

    /*
    **
    **
    */
    getRequestLog() {

        let remoteFamily = this.devToolSocket.getRemoteFamily();
        let remoteAddress = this.devToolSocket.getRemoteIp();
        let remotePort = this.request.socket.remotePort;
        let remoteSocket = '';

        if (remoteFamily === 'IPv6') {
            if (remoteAddress.indexOf('::ffff:') === 0)
                remoteSocket = `${Log.Dim}[::ffff:${Log.Reset}${Log.Bright}${remoteAddress.slice(7)}${Log.Reset}${Log.Dim}]:${remotePort}${Log.Reset}`;
            else
                remoteSocket = `${Log.Dim}[${Log.Reset}${Log.Bright}${remoteAddress}${Log.Reset}${Log.Dim}]:${remotePort}${Log.Reset}`;
        }
        else
            remoteSocket = `${Log.FgGreen}${Log.Bright}${remoteAddress}${Log.Reset}${Log.FgGreen}${Log.Dim}:${remotePort}${Log.Reset}`;

        let software = null;
        let ua = UAParser(this.request.headers['user-agent']);
        let os = ua.os ? ua.os.name : '';
        let browser = ua.browser ? `${ua.browser.name} ${ua.browser.major}` : '';
        if (os && browser)
            software = `${Log.Dim}${os} ${browser}${Log.Reset}`;
            
        if (software)
            return `${remoteSocket} ${software}`;
        else
            return `${remoteSocket}`;
    }
}