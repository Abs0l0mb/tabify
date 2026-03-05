'use strict';

import {
    Listener,
    TListenerTriggerCallback,
    TListenerOffCallback,
    Tools
} from '@src/classes';

export class Emitter {

    private listeners: {[id: string]: Listener} = {};

    constructor() {
    }

    /*
    **
    **
    */
    public on(event: string, triggerCallback: TListenerTriggerCallback) : Listener {

        const id = Tools.uid();

        const offCallback: TListenerOffCallback = () => {
            this.off(id);
        };

        const listener = new Listener(id, event, triggerCallback, offCallback);

        this.listeners[id] = listener;

        return listener;
    }

    /*
    **
    **
    */
    private off(id: string) : void {

        for (const listener of Object.values(this.listeners)) {

            if (listener && listener.getId() === id) {
                delete this.listeners[id];
                return;
            }
        }
    }

    /*
    **
    **
    */
    protected emit(event: string, data?: any) : void {

        for (const listener of Object.values(this.listeners)) {

            if (listener && listener.getEvent() === event)
                listener.trigger(data);
        } 
    }

    /*
    **
    **
    */
    protected clearListeners() : void {

        this.listeners = {};
    }
}