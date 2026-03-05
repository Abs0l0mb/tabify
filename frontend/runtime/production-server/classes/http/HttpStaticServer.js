'use strict';

import { createServer } from 'http';
import { readFileSync, existsSync, lstatSync, readdirSync } from 'fs';
import { resolve as _resolve, extname as _extname } from 'path';

import { Request } from './HttpRequest.js';
import { Response } from './HttpResponse.js';
import { RequestLogger } from './HttpRequestLogger.js';

import { Log } from '../Log.js';

export class HttpStaticServer {

    constructor() {

        this.server = createServer({
            IncomingMessage: Request,
            ServerResponse: Response
        }, this.onRequest.bind(this));

        this.staticDirectories = {};
    }

    /*
    **
    **
    */
    async onRequest(request, response) {

        try {

            if (request.method !== 'GET')
                return;
                
            request.decomposeUrl();

            new RequestLogger(request, response);

            if (!this.lookForStaticFile(request.builtUrl.pathname, response)) {
                
                if (!this.lookForStaticFile('/', response)) {

                    response.writeHead(404);
                    response.end('not found', 'utf-8');
                }
            }
        }
        catch(error) {

            console.log('[ httpStaticServer.onRequest Catched Error ]', error);
        }
    }

    /*
    **
    **
    */
    lookForStaticFile(path, response) {
        
        for (let directoryPath in this.staticDirectories) {

            for (let file of this.staticDirectories[directoryPath]) {

                if (file.path === path) {

                    if (!file.content)
                        file.content = readFileSync(file.filePath);

                    response.writeHead(200, {
                        'Content-Type': file.mime,
                        'Content-Length': file.content.byteLength
                    });

                    response.end(file.content, 'utf-8');

                    return true;
                }
            }
        }

        return false;
    }

    /*
    **
    **
    */
    getMime(extname) {

        return {
            '.html': 'text/html',
            '.js': 'text/javascript',
            '.png': 'image/png',
            '.css': 'text/css',
            '.ico': 'image/x-icon'
        }[extname];
    }

    /*
    **
    **
    */
    aspireDirectory(
        directoryPath,
        memoryLimit,
        memoryLimitPerFile,
        files = [],
        memoryUsed = 0,
        currentBasePath = '/') {
        
        if (!existsSync(directoryPath) || !lstatSync(directoryPath).isDirectory())
            return [];

        for (let entryName of readdirSync(directoryPath)) {
            
            let entryPath = _resolve(directoryPath, entryName);
            let entryStat = lstatSync(entryPath);

            if (entryStat.isFile()) {

                let mime = this.getMime(_extname(entryName));
                
                if (!mime)
                    continue;

                let content = null;

                if (memoryUsed + entryStat.size < memoryLimit && entryStat.size < memoryLimitPerFile) {
                    content = readFileSync(entryPath);
                    memoryUsed += entryStat.size;
                }

                if (entryName === 'index.html')
                    entryName = '';

                files.push({
                    path: `${currentBasePath}${entryName}`,
                    filePath: entryPath,
                    mime: mime,
                    content: content
                });
            }
            else if (entryStat.isDirectory()) {

                this.aspireDirectory(
                    entryPath, 
                    memoryLimit,
                    memoryLimitPerFile,
                    files, 
                    memoryUsed,
                    `${currentBasePath}${entryName}/`);
            }
        }

        return files;
    }

    /*
    **
    **
    */
    static(
        directoryPath,
        memoryLimit = 1000 * 1000 * 1000 /*1GB*/,
        memoryLimitPerFile = 1000 * 1000 * 5 /*5MB*/) {

        this.staticDirectories[directoryPath] = this.aspireDirectory(directoryPath, memoryLimit, memoryLimitPerFile);
    }

    /*
    **
    **
    */
    clearStatics() {

        this.staticDirectories = {};
    }

    /*
    **
    **
    */
    listen(port = 80, arg2 = undefined) {

        this.server.listen(port, arg2, function() {

            Log.green(`Http server listening on ${port}`);
        });
    }

    /*
    **
    **
    */
    async close() {

        if (!this.server || !this.server.listening)
            return;

        await new Promise(function(resolve) {
            this.server.close(resolve);
        }.bind(this));

        Log.green('Http server closed');
    }
}