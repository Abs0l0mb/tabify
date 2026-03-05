'use strict';

import {
    Div,
    Block,
    Menu,
    MenuItem,
    MenuItemData,
    ClientLocation,
    AltBox
} from '@src/classes';

export class Submenu extends Div {

    protected head: Div;
    protected body: Div;
    protected items: Div;

    private opened: boolean;

    constructor(protected data: MenuItemData, protected menu: Menu, parent: Block) {

        super('sub-menu', parent);

        if (!this.data.submenu)
            return;

        if (this.data.extraClass)
            this.addClass(this.data.extraClass);

        this.head = new Div('head', this);
        this.head;

        new Div('text', this.head).write(data.label);
        
        new AltBox(this.head, this.data.label, {
            className: 'navigation-alt-box'
        });

        this.body = new Div('body', this);
        this.items = new Div('items', this.body);

        this.buildItem();

        this.head.onNative('click', this.toggle.bind(this));
        
        if (this.data.submenuOpened)
            this.open();
        else
            this.close();

        ClientLocation.get().router.on('beforeRoute', (path: string) => {
            for (const submenuItemData of this.data.submenu!) {
                if (path.split('?')[0] === submenuItemData.path) {
                    this.select();
                    break;
                }
                else
                    this.unselect();
            }
        });

        if (this.items.element.children.length === 0)
            this.delete();
    }

    /*
    **
    **
    */
    protected buildItem() : void {

        if (!this.data.submenu)
            return;

        for (const submenuItemData of this.data.submenu)
            new MenuItem(submenuItemData, this.menu, this.items);
    }

    /*
    **
    **
    */
    private select() : void {

        this.setData('selected', 1);
    }

    /*
    **
    **
    */
    public unselect() : void {

        this.setData('selected', 0);
    }

    /*
    **
    **
    */
    protected toggle() : void {

        if (!this.opened)
            this.open();
        else
            this.close();
    }

    /*
    **
    **
    */
    protected open() : void {

        this.opened = true;

        this.emit('open', this);
        
        this.setData('opened', 1);

        this.body.setStyle('height', `${this.items.element.offsetHeight}px`);
    }

    /*
    **
    **
    */
    protected close() : void {

        this.opened = false;

        this.setData('opened', 0);

        this.body.setStyle('height', `0px`);
    }
}