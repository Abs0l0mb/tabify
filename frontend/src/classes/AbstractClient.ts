'use strict';

import {
    Router,
    Api,
    FontManager,
    Block,
    Div,
    Tools,
    Emitter,
    ClientLocation
} from '@src/classes';

export interface ClientConfig {
    title?: string,
    fonts?: string[] 
};

export interface PointerPosition {
    x: number, 
    y: number
}

declare const __APP_NAME__: string;

export abstract class AbstractClient extends Emitter {
    
    public title: string = __APP_NAME__;
    public fontManager: FontManager;
    public block: Block;
    public firstScreen: Block;
    public contentRoot: Block;
    public router: Router;
    public api: Api;
    public visibilityChangeListener: any;
    public parameters: any;
    public localURLSCache: any = {};
    public dataCache: any = {};
    public tools: any = Tools;
    public pointer: PointerPosition;

    constructor(private config: ClientConfig) {

        super();

        let parametersElement = document.getElementById('parameters');
        
        if (parametersElement)
            this.parameters = JSON.parse(parametersElement.innerText);
        else
            this.parameters = {};

        document.addEventListener('deferredResourcesLoaded', this.onDeferredResourcesLoaded.bind(this));
        document.addEventListener('click', this.onDOMClick.bind(this));
        document.addEventListener('mousemove', this.onMouseMove.bind(this));
        document.addEventListener('mouseup', this.onMouseUp.bind(this));

        ClientLocation.set(this);
    }

    /*
    **
    **
    */
    private async onDeferredResourcesLoaded() : Promise<void> {

        await this.beforeInit();

        if (this.config.fonts) {

            this.fontManager = new FontManager(this.config.fonts);
            this.fontManager.on('load', this.init.bind(this));
        }
        else
            this.init();
    }

    /*
    **
    **
    */
    private async onDOMClick(event: MouseEvent) : Promise<void> {

        this.emit('document-click', {
            target: event.target,
            x: event.pageX,
            y: event.pageY
        });
    }

    /*
    **
    **
    */
    private async onMouseMove(event: MouseEvent) : Promise<void> {

        this.pointer = {
            x: event.clientX,
            y: event.clientY
        };
        
        this.emit('mouse-move', this.pointer);
    }

    /*
    **
    **
    */
    private async onMouseUp(event: MouseEvent) : Promise<void> {
        
        this.emit('mouse-up');
    }

    /*
    **
    **
    */
    public async init() : Promise<void> {

        document.addEventListener('scroll', this.onScroll.bind(this));
        window.addEventListener('resize', this.onResize.bind(this));
        this.initVisibilityChangeListener();
        this.onResize();
    
        this.block = new Block(document.getElementById('app'));
        this.block.setData('mobile', Tools.isMobile() ? 1 : 0);

        this.firstScreen = new Block(document.getElementById('first-screen'));  
        this.contentRoot = new Div('content-root', this.block);
        
        this.router = new Router();

        this.api = new Api();

        await this.afterInit();

        this.api.on('not-connected', this.onNotConnected.bind(this));
        this.api.on('connected', this.onConnected.bind(this));
        this.api.checkAuth();
    }

    /*
    **
    **
    */
    public ready() : void {

        this.block.setData('ready', 1);
    }

    /*
    **
    **
    */
    abstract onNotConnected(data: any) : void
    
    /*
    **
    **
    */
    abstract onConnected(data: any) : void

    /*
    **
    **
    */
    private initVisibilityChangeListener() : void {

        if (this.visibilityChangeListener)
            return;

        this.visibilityChangeListener = document.addEventListener('visibilitychange', this.onVisibilityChange.bind(this), false);
    }

    /*
    **
    **
    */
    private onVisibilityChange() : void {

        this.emit('visibilityChange', document.visibilityState === "visible");
    }

    /*
    **
    **
    */
    private onScroll(event): void {

        this.emit('scroll', event);
    }

    /*
    **
    **
    */
    private onResize() : void {

        this.emit('resize');
    }
    
    /*
    **
    **
    */
    abstract beforeInit() : Promise<void>

    /*
    **
    **
    */
    abstract afterInit() : Promise<void>
}