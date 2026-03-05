
'use strict';

import {
    InputStructure,
    Block,
    Div,
    Tools
} from '@src/classes';

export interface SelectItem {
    label: string;
    value: number | string | null;
}

export interface SelectInputData {
    label: string;
    value?: number | string;
    mandatory?: boolean;
    items?: SelectItem[];
    class?: string;
}

export class SelectInput extends InputStructure {

    private focusInput: Block;
    private inputLike: Div;
    private itemsContainer: Div;
    private items: SelectItem[] = [];
    private value: string | number | null = null;
    private text: string = '';
    private displayTimeouts: any[] = [];
    private isOver: boolean = false;
    private isFocused: boolean = false;

    constructor(public data: SelectInputData, parent?: Block) {

        super(data, parent);

        this.setCustomType('select');

        this.focusInput = new Block('input', {}, this.inputContainer)
            .onNative('mousedown', this.onMouseDown.bind(this))
            .onNative('focusin', this.onFocusIn.bind(this))
            .onNative('focusout', this.onFocusOut.bind(this))
            .onNative('keydown', this.onKeyDown.bind(this));

        this.onNative('mouseover', this.onMouseOver.bind(this));
        this.onNative('mouseout', this.onMouseOut.bind(this));

        this.inputLike = new Div('input-like', this.inputContainer);
        
        this.itemsContainer = new Div('items-container', this);
        
        if (this.data.items)
            this.setItems(this.data.items);
        
        if (this.data.value)
            this.setValue(this.data.value);
    }

    /*
    **
    **
    */
    public setValue(value: string | number | null, emit: boolean = false) : void {

        if (!this.items)
            return;

        for (let data of this.items) {

            if (data.value === value) {

                this.value = value;
                this.text = data.label;
                
                if (emit)
                    this.emit('value', this.value);

                this.inputLike.write(data.label);
                this.setFilled(true);

                break;
            }
        }
    }

    /*
    **
    **
    */
    public getValue() : string | number | null {

        return this.value;
    }

    /*
    **
    **
    */
    public getText() : string {

        return this.text;
    }

    /*
    **
    **
    */
    public setItems(items: SelectItem[]) : void {

        items.unshift({
            label: '',
            value: null
        });

        this.items = items;

        this.itemsContainer.empty();

        for (let data of items) {
            
            new Div({}, this.itemsContainer)
                .write(data.label)
                .onNative('mousedown', () => {
                    this.setValue(data.value, true);
                    this.focusInput.element.blur();
                    this.hideItems();
                });
        }
    }

    /*
    **
    **
    */
    private async displayItems() : Promise<void> {

        this.clearDisplayTimeouts();

        const initialScrollHeight = this.element.parentElement.scrollHeight;

        this.setData('items-display-state', 'started');

        this.displayTimeouts.push(setTimeout(() => {
            
            this.setData('items-display-state', 'displayed');
            
            this.displayTimeouts.push(setTimeout(() => {
                if (this.element.parentElement.scrollHeight - initialScrollHeight > 0)
                    this.element.parentElement.scrollTop = this.element.parentElement.scrollHeight;
            }, 150));

        }, 50));
    }

    /*
    **
    **
    */
    private async hideItems() : Promise<void> {

        this.clearDisplayTimeouts();
        
        this.setData('items-display-state', 'hidden');
    }

    /*
    **
    **
    */
    private clearDisplayTimeouts() : void {

        for (let timeout of this.displayTimeouts)
            clearTimeout(timeout);
    }

    /*
    **
    **
    */
    private onMouseOver() : void {

        this.isOver = true;
    }

    /*
    **
    **
    */
    private onMouseOut() : void {

        this.isOver = false;
    }

    /*
    **
    **
    */
    private async onMouseDown() : Promise<void> {
        
        if (!this.isFocused) {
            this.isFocused = true;
            this.displayItems();
        }
        else {
            await Tools.sleep(1);
            this.focusInput.element.blur();
            this.hideItems();
        }
    }

    /*
    **
    **
    */
    private async onFocusIn(event: Event) : Promise<void> {
        
        this.displayItems();
        this.isFocused = true;
    }

    /*
    **
    **
    */
    private onFocusOut() : void {

        this.isFocused = false;

        if (!this.isOver) 
            this.hideItems();
    }

    /*
    **
    **
    */
    protected onKeyDown(event: KeyboardEvent) : void {

        if (event.key === 'Tab')
            event.preventDefault();
    }
}