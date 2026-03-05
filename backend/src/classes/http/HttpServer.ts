'use strict';

import {
    HttpRequest,
    HttpResponse,
    HttpRequestLog,
    WebSocketClient,
    PublicError,
    Log,
    Emitter
} from '@src/classes';

import {
    WebSocketServer,
    WebSocket
} from 'ws';

import * as http from 'http';
import * as net from 'net';

export type MiddlewareCallback = (request: HttpRequest, response: HttpResponse, next: () => void) => Promise<void>;
export type UpgradeMiddlewareCallback = (request: HttpRequest, next: () => void) => Promise<void>; 
export type EndpointCallback = (request: HttpRequest, response: HttpResponse) => Promise<void>;
export type UpgradeEndpointCallback = (client: WebSocketClient) => Promise<void>;

export interface Middleware {
    pattern: string,
    callback: MiddlewareCallback,
    method?: string | string[]
}

export interface UpgradeMiddleware {
    pattern: string,
    callback: UpgradeMiddlewareCallback
}

export interface Endpoint {
    method: string,
    pattern: string,
    wildcard?: boolean,
    callback: EndpointCallback
}

export interface UpgradeEndpoint {
    pattern: string,
    callback: UpgradeEndpointCallback
}

export class HttpServer extends Emitter {

    private wrapped: http.Server;
    private middlewares: Middleware[] = [];
    private upgradeMiddlewares: UpgradeMiddleware[] = [];
    private endpoints: Endpoint[] = [];
    private upgradeEndpoints: UpgradeEndpoint[] = [];

    protected webSocketServer: WebSocketServer = new WebSocketServer({
        noServer: true
    });

    constructor() {

        super();
        
        this.wrapped = http.createServer({
            IncomingMessage: HttpRequest,
            ServerResponse: HttpResponse
        } as any, this.onRequest.bind(this));

        this.wrapped.on('upgrade', this.onUpgrade.bind(this));
    }

    /*
    **
    **
    */
    protected async onRequest(request: HttpRequest, response: HttpResponse) : Promise<void> {

        try {
            
            request.buildHeaders();
            request.buildURL();

            await request.extractGETParameters();
            await request.extractPOSTParameters();
            
            new HttpRequestLog(request, response);

            await this.processMiddlewares(request, response);

            if (response.isFinished()) return;
            if (await this.lookForEndpoint(request, response)) return;
            if (response.isFinished()) return;
            if (await this.lookForWildcardEndpoint(request, response)) return;
            if (response.isFinished()) return;

            throw new PublicError('endpoint-not-found');
        }
        catch(error) {

            if (typeof error === 'object'
             && error !== null
             && error.public
             && error.message) {
                response.sendErrorContent(error.message);
            }
            else {

                response.sendErrorContent('internal-error');

                Log.red(`ERROR on ${request.getURL().pathname}`);
                Log.printError(error);
            }
        }
    }

    /*
    **
    **
    */
    protected async onUpgrade(request: HttpRequest, socket: net.Socket, head: Buffer) : Promise<void> {
        
        try {

            request.buildHeaders();
            request.buildURL();
            
            request.wsAuthPatch();

            await this.processUpgradeMiddlewares(request);

            await this.lookForUpgradeEndpoint(request, socket, head);
            
        } catch (error) {

            Log.red(`ERROR during WebSocket upgrade on ${request.getURL().pathname}`);
            Log.printError(error);
    
            socket.write('HTTP/1.1 409 Conflict\r\n\r\n');
            socket.destroy();
        }
    }

    /*
    **
    **
    */
    private async processMiddlewares(request: HttpRequest, response: HttpResponse) : Promise<void> {

        for (let middleware of this.middlewares) {

            if (request.getURL().pathname === middleware.pattern) {
                
                if (typeof middleware.method === 'string' && request.method !== middleware.method)
                    continue;
                
                else if (Array.isArray(middleware.method) && request.method && !middleware.method.includes(request.method))
                    continue;
                    
                await new Promise(async function(resolve: any, reject: any) {
                    try {
                        await middleware.callback(request, response, resolve);
                    } catch(error) {
                        reject(error);
                    }
                });
            }
        }
    }

    /*
    **
    **
    */
    private async processUpgradeMiddlewares(request: HttpRequest) : Promise<void> {

        for (const middleware of this.upgradeMiddlewares) {

            if (request.getURL().pathname.indexOf(middleware.pattern) === 0) {
                    
                await new Promise(async function(resolve: any, reject: any) {
                    try {
                        await middleware.callback(request, resolve);
                    } catch(error) {
                        reject(error);
                    }
                });
            }
        }
    }

    /*
    **
    **
    */
    private async lookForEndpoint(request: HttpRequest, response: HttpResponse) : Promise<boolean> {

        for (const endpoint of this.endpoints) {

            if (request.method === endpoint.method 
             && request.getURL().pathname === endpoint.pattern) {

                await endpoint.callback(request, response);
                
                return true;
            }
        }

        return false;
    }

    /*
    **
    **
    */
    private async lookForWildcardEndpoint(request: HttpRequest, response: HttpResponse) : Promise<boolean> {

        for (const endpoint of this.endpoints) {

            if (request.method === endpoint.method 
             && request.getURL().pathname.indexOf(endpoint.pattern) === 0 
             && endpoint.wildcard) {

                await endpoint.callback(request, response);
                
                return true;
            }
        }

        return false;
    }

    /*
    **
    **
    */
    private async lookForUpgradeEndpoint(request: HttpRequest, socket: net.Socket, head: Buffer) : Promise<boolean> {

        for (const endpoint of this.upgradeEndpoints) {

            if (request.getURL().pathname === endpoint.pattern) {

                const webSocket: WebSocket | null = await this.handleWebSocketUpgrade(request, socket, head);

                if (webSocket)
                    await endpoint.callback(new WebSocketClient(webSocket, request));
                
                return true;
            }
        }

        return false;
    }

    /*
    **
    **
    */
    private addMiddleware(pattern: string, callback: MiddlewareCallback, method?: string | string[]) : void {

        this.middlewares.push({
            pattern: pattern,
            callback: callback,
            method: method
        });
    }

    /*
    **
    **
    */
    private addUpgradeMiddleware(pattern: string, callback: UpgradeMiddlewareCallback) : void {

        this.upgradeMiddlewares.push({
            pattern: pattern,
            callback: callback
        });
    }

    /*
    **
    **
    */
    private addEndpoint(method: string, pattern: string, callback: EndpointCallback) : void {

        let wildcard = false;
        
        if (pattern.slice(-1) === '*') {
            wildcard = true;    
            pattern = pattern.slice(0, -1);
        }

        if (pattern.slice(-1) === '/')
            pattern = pattern.slice(0, -1);

        this.endpoints.push({
            method: method,
            pattern: pattern,
            wildcard: wildcard,
            callback: callback
        });
    }

    /*
    **
    **
    */
    public use(pattern: string, middleware: MiddlewareCallback | MiddlewareCallback[], method?: string | string[]) : HttpServer {
        
        if (Array.isArray(middleware)) {
            for (const middleware_ of middleware)
                this.use(pattern, middleware_);
            return this;
        }

        this.addMiddleware(pattern, middleware, method);
        
        return this;
    }

    /*
    **
    **
    */
    public upgradeUse(pattern: string, middleware: UpgradeMiddlewareCallback | UpgradeMiddlewareCallback[], method?: string | string[]) : HttpServer {
        
        if (Array.isArray(middleware)) {
            for (const middleware_ of middleware)
                this.upgradeUse(pattern, middleware_);
            return this;
        }

        this.addUpgradeMiddleware(pattern, middleware);
        
        return this;
    }

    /*
    **
    **
    */
    public options(pattern: string, callback: EndpointCallback, middleware?: MiddlewareCallback | MiddlewareCallback[]) : HttpServer {
        
        const method = 'OPTIONS';

        this.addEndpoint(method, pattern, callback);

        if (middleware)
            this.use(pattern, middleware, method);

        return this;
    }

    /*
    **
    **
    */
    public get(pattern: string, callback: EndpointCallback, middleware?: MiddlewareCallback | MiddlewareCallback[]) : HttpServer {

        const method = 'GET';

        this.addEndpoint(method, pattern, callback);

        if (middleware)
            this.use(pattern, middleware, method);

        return this;
    }

    /*
    **
    **
    */
    public post(pattern: string, callback: EndpointCallback, middleware?: MiddlewareCallback | MiddlewareCallback[]) : HttpServer {

        const method = 'POST';

        this.addEndpoint(method, pattern, callback);

        if (middleware)
            this.use(pattern, middleware, method);

        return this;
    }

    /*
    **
    **
    */
    public upgrade(pattern: string, callback: UpgradeEndpointCallback, middleware?: UpgradeMiddlewareCallback | UpgradeMiddlewareCallback[]) : HttpServer {

        this.upgradeEndpoints.push({
            pattern: pattern,
            callback: callback
        });

        if (middleware)
            this.upgradeUse(pattern, middleware, 'GET');

        return this;
    }

    /*
    **
    **
    */
    private async handleWebSocketUpgrade(request: HttpRequest, socket: net.Socket, head: Buffer) : Promise<WebSocket | null> {

        return new Promise<WebSocket>(resolve => {
            this.webSocketServer.handleUpgrade(request, request.socket, head, (webSocket: WebSocket) => {
                resolve(webSocket);
            });
        });
    }

    /*
    **
    **
    */
    public listen(port: number = 8080, host?: string) : void {

        this.wrapped.listen(port, host, function() {
            Log.green(`Http server listening on ${port}`);
        });
    }

    /*
    **
    **
    */
    public broadcast(data: any) : void {

        this.webSocketServer.clients.forEach(function each(client) {

            if (client.readyState === WebSocket.OPEN) {

                client.send(JSON.stringify(data), {
                    binary: false
                });
            }
        });
    }
}