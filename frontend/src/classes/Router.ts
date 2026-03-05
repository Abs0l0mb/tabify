'use strict';

import {
    Emitter,
    ApiRequestsManager,
    Page,
    ClientLocation,
    Tools
} from '@src/classes';

export interface RoutesConfig {
    [path: string]: typeof Page
}

export type RouteParams = {[key: string]: string | number};

export class Router extends Emitter {

    private routes: RoutesConfig = {};
    private path: string;
    private page: Page;
    private params: RouteParams = {};

    constructor() {

        super();
        
        window.onpopstate = this.onPopState.bind(this);
    }

    /*
    **
    **
    */
    private async onPopState() : Promise<void> {

        this.routeCurrentPath(false);
    }

    /*
    **
    **
    */
    public async setRoutes(routes: RoutesConfig) : Promise<void> {

        this.routes = routes;
    }

    /*
    **
    **
    */
    public async route(path: string, pushState: boolean = true) : Promise<void> {

        const rootPath = ClientLocation.get().rootPath;
        
        if (path.slice(0, rootPath.length) !== rootPath)
            path = rootPath + path;

        this.emit('beforeRoute', path);
        
        ApiRequestsManager.clearPendingRequests();
    
        let pageConstructor = this.routes[path.slice(rootPath.length).split('?')[0]];

        if (!pageConstructor)
            throw new Error(`Unable to found a page constructor for the specified path ${path}`);
        
        if (pushState && path !== location.pathname + location.search)
            this.pushState(path);

        if (this.page) {
            await this.page.onLeave();
            this.page.delete();
        }

        this.params = Router.getParams(path);

        this.path = path;
        this.page = new pageConstructor();

        ClientLocation.get().contentRoot.append(this.page);

        Tools.scrollTop();

        this.emit('route', path);
    }

    /*
    **
    **
    */
    public setParams(params: {[key: string]: string | number}) : void {

        for (const [key, value] of Object.entries(params))
            this.setParam(key, value);
    }

    /*
    **
    **
    */
    public setParam(key: string, value: string | number) : void {

        this.params[key] = value;
    }

    /*
    **
    **
    */
    public getParams() : RouteParams {

        return this.params;
    }

    /*
    **
    **
    */
    public routeCurrentPath(pushState: boolean = true) : void {

        for (let path of Object.keys(this.routes)) {
            if (ClientLocation.get().rootPath + path === location.pathname) {
                this.route(location.pathname + location.search, pushState);
                return;
            }
        }

        this.routeFirstPath();
    }

    /*
    **
    **
    */
    public routeFirstPath(pushState: boolean = true) : void {

        this.route(Object.keys(this.routes)[0], pushState);
    }

    /*
    **
    **
    */
    static getParams(url: string) : RouteParams {
        
        if (url.indexOf('?') === -1)
            return {};

        return url.split('?')[1].split("&").reduce(function(prev, curr, i, arr) {
            const part = curr.split("=");
            let stringValue: string = decodeURIComponent(part[1]);
            let value: string | number = stringValue;
            if (!isNaN(Number(value)))
                value = Number(value);
            prev[decodeURIComponent(part[0])] = value;
            return prev;
        }, {});
    }

    /*
    **
    **
    */
    public pushParamsState() : void {

        if (!this.path)
            return;

        let path = this.path.split('?')[0];
    
        const query = Object.keys(this.params).map(key => {
            return `${key}=${encodeURIComponent(this.params[key])}`;
        }).join('&');

        path = `${path}?${query}`;

        this.pushState(path);
    }

    /*
    **
    **
    */
    private pushState(path: string) : void {

        window.history.pushState({}, '', path); 

        this.emit('state-pushed');
    }
}